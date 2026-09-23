"""DSV41_ENGRAM_DISK host path vs vLLM 0.30's _engram_lookup_kernel.

Writes a small fp8/ue8m0 table as a real safetensors shard (plus an index),
then for a rank slice [vocab_start, vocab_end) and a head window compares the
kernel output (GPU, table resident) with DiskEngramTable + the lookup logic of
the patched ParallelEngramEmbedding. Covers padded heads (head >= TOTAL_HEADS),
DEAD_ID (-1), ids owned by other ranks and duplicate ids. Must be bit-exact.
"""

import json
import os
import sys
import tempfile

import torch
from safetensors.torch import save_file

os.environ["DSV41_ENGRAM_DISK"] = "1"

from vllm.models.deepseek_v41.common.engram import _engram_lookup_kernel  # noqa: E402
from vllm.models.deepseek_v41.nvidia.engram_disk import (  # noqa: E402
    DiskEngramTable,
    gather_dequant,
)

torch.manual_seed(0)
ROWS, DIM, QB = 5000, 256, 32
LAYER = 14
TOTAL_HEADS, LOCAL_HEADS = 7, 4  # 2 ranks: rank 1 gets heads 4..7, one padded
failures = 0

d = tempfile.mkdtemp()
w = (torch.randn(ROWS, DIM) * 3).to(torch.float8_e4m3fn)
s = torch.randint(110, 140, (ROWS, DIM // QB), dtype=torch.uint8).view(
    torch.float8_e8m0fnu
)
save_file(
    {
        f"layers.{LAYER}.engram.embed.weight": w,
        f"layers.{LAYER}.engram.embed.scale": s,
        "pad": torch.zeros(3),
    },
    os.path.join(d, "shard.safetensors"),
)
with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
    json.dump(
        {
            "weight_map": {
                f"layers.{LAYER}.engram.embed.weight": "shard.safetensors",
                f"layers.{LAYER}.engram.embed.scale": "shard.safetensors",
            }
        },
        f,
    )

for head_start, vocab_start, vocab_end in [(0, 0, 2600), (4, 2600, 5000)]:
    T = 37
    ids = torch.randint(0, ROWS, (T, TOTAL_HEADS), dtype=torch.int64)
    ids[3, :] = -1  # DEAD_ID
    ids[5, :] = ids[6, :]  # duplicates
    ids = ids.to(torch.int32).cuda()

    # Kernel reference, table resident on the GPU.
    wt = w[vocab_start:vocab_end].cuda()
    st = s.view(torch.uint8)[vocab_start:vocab_end].cuda()
    ref = torch.empty(T, LOCAL_HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    rows = T * LOCAL_HEADS
    grid = min(-(-rows // 16), 8)
    _engram_lookup_kernel[(grid,)](
        wt, st, ids, ref, vocab_start, vocab_end, rows,
        ids.stride(0), ids.stride(1),
        HEAD_START=head_start, LOCAL_HEADS=LOCAL_HEADS, TOTAL_HEADS=TOTAL_HEADS,
        DIM=DIM, QUANT_BLOCK=QB, BLOCK_R=16, GRID=grid,
    )

    # Disk path: same steps as the patched ParallelEngramEmbedding.lookup.
    table = DiskEngramTable(d, LAYER, DIM, QB, vocab_start, vocab_end - vocab_start)
    head_end = min(head_start + LOCAL_HEADS, TOTAL_HEADS)
    x = ids[:, head_start:head_end].to("cpu", dtype=torch.int64)
    if x.shape[1] < LOCAL_HEADS:
        x = torch.cat(
            (x, torch.full((T, LOCAL_HEADS - x.shape[1]), -1, dtype=torch.int64)), 1
        )
    x = x.reshape(-1)
    owned = (x >= vocab_start) & (x < vocab_end)
    rel = torch.where(owned, x - vocab_start, 0)
    got = gather_dequant(table, rel, owned).view(T, LOCAL_HEADS, DIM).cuda()

    exact = torch.equal(got, ref)
    nz = int((ref != 0).any(-1).sum())
    print(f"head_start={head_start} rows [{vocab_start},{vocab_end}): "
          f"bit-exact={exact} nonzero_rows={nz}/{rows} "
          f"max_abs_diff={(got.float() - ref.float()).abs().max().item()}")
    failures += not exact

print("PASS" if failures == 0 else "FAIL")
sys.exit(1 if failures else 0)
