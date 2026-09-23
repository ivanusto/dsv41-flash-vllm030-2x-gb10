# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk-backed Engram tables for unified-memory GPUs (DGX Spark / GB10).

On GB10 "pinned host memory" is the same pool the GPU allocates from, so the
CPU-offload path does not save anything. With ``DSV41_ENGRAM_DISK=1`` each
rank reads only the rows a step needs straight from the safetensors shards
with positional reads on a thread pool, dequantizes them on the CPU
(fp8 e4m3 x ue8m0 block scales -> bf16, same math as the lookup kernel) and
copies them into the staging buffer. The forward then needs a host round trip,
so this path requires ``--enforce-eager``.

Ported to vLLM 0.30 from the Tech2Wild/Kai patch (tonyd2wild
DeepSeek-V4.1-Flash-vLLM-DGX-Spark, 2026-09-10), eager path only.
"""

import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENGRAM_DISK = os.environ.get("DSV41_ENGRAM_DISK", "0") == "1"
_THREADS = int(os.environ.get("DSV41_ENGRAM_DISK_THREADS", "32"))
_CHUNK = int(os.environ.get("DSV41_ENGRAM_DISK_CHUNK", "16"))
_POOL: ThreadPoolExecutor | None = None


def is_engram_table(name: str) -> bool:
    return ENGRAM_DISK and name.endswith(
        (".engram.embed.weight", ".engram.embed.scale")
    )


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(
            max_workers=_THREADS, thread_name_prefix="engram-disk"
        )
    return _POOL


def _pread_rows(
    fd: int, base: int, rel: list[int], lo: int, hi: int, row_bytes: int, buf
) -> None:
    for i in range(lo, hi):
        off = base + rel[i] * row_bytes
        view = buf[i * row_bytes : (i + 1) * row_bytes]
        got = 0
        while got < row_bytes:
            n = os.preadv(fd, [view[got:]], off + got)
            if n <= 0:
                raise OSError("engram disk table: short read")
            got += n


def _parallel_read(jobs: list) -> None:
    """jobs: [(fd, base, rel, row_bytes, memoryview)]. Only the calling thread
    submits, so the shared pool cannot deadlock."""
    total = sum(len(job[2]) for job in jobs)
    if total == 0:
        return
    if total == 1:
        for fd, base, rel, row_bytes, buf in jobs:
            _pread_rows(fd, base, rel, 0, len(rel), row_bytes, buf)
        return
    chunk = max(1, min(_CHUNK, -(-total // _THREADS)))
    futs = [
        _pool().submit(
            _pread_rows, fd, base, rel, lo, min(lo + chunk, len(rel)), row_bytes, buf
        )
        for fd, base, rel, row_bytes, buf in jobs
        for lo in range(0, len(rel), chunk)
    ]
    for fut in futs:
        fut.result()


class DiskEngramTable:
    """One layer's rank-local rows, read from the checkpoint on demand."""

    def __init__(
        self,
        model_dir: str,
        layer_id: int,
        dim: int,
        block_size: int,
        row_start: int,
        num_rows: int,
    ) -> None:
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        wname = f"layers.{layer_id}.engram.embed.weight"
        sname = f"layers.{layer_id}.engram.embed.scale"
        self.w_fd, w_off, w_shape = self._open(model_dir, weight_map[wname], wname)
        self.s_fd, s_off, s_shape = self._open(model_dir, weight_map[sname], sname)
        self.dim = dim
        self.sb = dim // block_size
        assert w_shape[1] == dim, (w_shape, dim)
        assert s_shape[1] == self.sb, (s_shape, self.sb)
        assert s_shape[0] == w_shape[0], (s_shape, w_shape)
        # The checkpoint holds every rank's hash heads; callers pass rank-local
        # row ids, so reads start at this rank's first row.
        assert 0 <= row_start and row_start + num_rows <= w_shape[0], (
            row_start,
            num_rows,
            w_shape,
        )
        self.w_off = w_off + row_start * dim
        self.s_off = s_off + row_start * self.sb
        logger.info(
            "Engram DISK: layer %d rows [%d, %d) from %s, %.2f GiB not allocated",
            layer_id,
            row_start,
            row_start + num_rows,
            weight_map[wname],
            num_rows * (dim + self.sb) / 1024**3,
        )

    @staticmethod
    def _open(model_dir: str, fname: str, tname: str) -> tuple[int, int, tuple]:
        path = os.path.join(model_dir, fname)
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        except OSError:
            pass
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n))[tname]
        assert meta["dtype"] in ("F8_E4M3", "F8_E8M0", "U8"), meta["dtype"]
        return fd, 8 + n + meta["data_offsets"][0], tuple(meta["shape"])

    def read_jobs(self, rel: list[int], w: torch.Tensor, s: torch.Tensor) -> list:
        return [
            (self.w_fd, self.w_off, rel, self.dim, memoryview(w.numpy()).cast("B")),
            (self.s_fd, self.s_off, rel, self.sb, memoryview(s.numpy()).cast("B")),
        ]

    def dequant(self, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """fp8 e4m3 rows (as uint8) x ue8m0 block scales -> [R, dim] fp32."""
        rows = w.shape[0]
        vals = (
            w.view(torch.float8_e4m3fn)
            .to(torch.float32)
            .view(rows, self.sb, self.dim // self.sb)
        )
        # The ue8m0 byte is the fp32 exponent field: 2^(e-127).
        scale = (s.to(torch.int32) << 23).view(torch.float32)
        return (vals * scale[:, :, None]).reshape(rows, self.dim)


def gather_dequant(
    table: DiskEngramTable, rel: torch.Tensor, owned: torch.Tensor
) -> torch.Tensor:
    """rel: [R] int64 CPU rank-local rows, owned: [R] bool -> [R, dim] bf16 CPU.
    Rows are de-duplicated before reading; unowned rows are zero."""
    uniq, inverse = torch.unique(rel, return_inverse=True)
    w = torch.empty((uniq.numel(), table.dim), dtype=torch.uint8)
    s = torch.empty((uniq.numel(), table.sb), dtype=torch.uint8)
    _parallel_read(table.read_jobs(uniq.tolist(), w, s))
    out = table.dequant(w, s)[inverse]
    out[~owned] = 0
    return out.to(torch.bfloat16)
