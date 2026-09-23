# DeepSeek-V4.1-Flash REAP-256E on stock vLLM 0.30.0, two GB10 (TP=2)

A port of the [Libertai two-Spark recipe](https://github.com/Libertai/dsv41-flash-vllm-2x-spark)
from the 2026-09-11 vLLM nightly it was written against to the official
**vLLM 0.30.0** release image, verified on two GB10 boxes (128 GB unified memory
each) connected over RoCE v2.

vLLM 0.30.0 ships `DeepseekV41ForCausalLM`, but the stock release still cannot
serve this model across two GB10 on its own: FlashInfer has no sparse-MLA
instantiation for the head count TP=2 produces, and the SM12x compressed-page
geometry still needs a fix ([vLLM #57028](https://github.com/vllm-project/vllm/pull/57028),
unmerged at the time of writing). This repository is what closes that gap:
nine modified vLLM files, one locally compiled kernel, and the tests that say
whether it is right.

Measured on 2026-09-23: loads in about 69 minutes, KV pool 372,529 tokens,
32K context, single-stream decode 9 to 10 tok/s, 31.5 tok/s aggregate at 8
concurrent requests. Correctness checks all pass (finite logprobs, 26K needle,
tool calls, reasoning, Traditional Chinese with zero Simplified characters).
See [RESULTS.md](RESULTS.md) for the conditions behind those numbers, including
what was not measured.

## What is here

```
files/      the nine files mounted over the image at run time: eight modified
            vLLM files plus engram_disk.py, which is new
patches/    the eight modified files as unified diffs against stock 0.30.0
scripts/    serve.sh, the two-node launcher
tests/      bit-exact equivalence test for the disk-backed Engram path
docs/       porting notes, written up in Traditional Chinese
```

The long-form write-up of how this port was done, and why each piece was kept,
rewritten or dropped, is in [docs/porting-notes-zh-TW.md](docs/porting-notes-zh-TW.md).

## The five things that had to change

1. **Renamed module tree.** 0.30 moved `models/deepseek_v4_1/` to
   `models/deepseek_v41/` and split Engram into `common/engram.py` and
   `nvidia/engram.py`, so whole-file mounts from the old recipe either miss or
   silently revert upstream fixes. Everything here is expressed as a diff.
2. **Engram on disk, rewritten.** Weights are about 100 GiB per rank and the
   Engram tables another 47 GiB; on unified memory "offload to host RAM" saves
   nothing, so the tables stay in the checkpoint and rows are read per step.
   In 0.30 this fits as an option on the NVIDIA embedding subclass
   (`disk_source`), leaving the shared path untouched. Enable with
   `DSV41_ENGRAM_DISK=1`; it needs `--enforce-eager` and raises a clear error if
   it is reached under CUDA graph capture.
3. **Loader filter instead of a weight_utils patch.** Skipping the Engram
   tensors is done in `should_skip_weight`, which every safetensors iterator in
   0.30 calls, rather than in one iterator.
4. **One patch dropped.** The old top-k override is now a supported option:
   `--kernel-config '{"sparse_indexer_topk_backend": "per_row"}'`.
5. **FlashInfer instantiation.** TP=2 needs `(32, 1152)` and `(32, 640)` sparse
   MLA kernels, which the wheel does not ship, and the prebuilt AOT module makes
   the JIT path skip your patched source entirely. Use the upstream recipe's
   `patches/flashinfer-dsv41-sm120-tp2.patch` with its `make-fipatch.sh` and
   `build-fi.sh` (mask the AOT directory, then check the symbol exists with
   `nm`). A build that "succeeds" without the symbol means the patch never
   compiled.

Also taken: the MXFP8 empty-batch guards and the multi-stream capture fallback
from vLLM #57028. Its attention hunk duplicates the recipe's compressed-page
fix and was not taken.

One warning from doing this: applying the old diffs with fuzz put a two-line
hunk inside another function's `return` statement. It happened to be a syntax
error; a slightly different landing spot would have been valid Python with the
wrong meaning. Review every fuzzy hunk by hand and compile afterwards.

## Correctness of the disk path

`tests/test_engram_disk_equiv.py` writes a real safetensors shard, then compares
the host gather and dequantization against 0.30's own `_engram_lookup_kernel`
for two rank slices, padded heads, `DEAD_ID`, out-of-range ids and duplicate
ids. The bar is **bit-exact**, not "close enough": this is a table lookup with
no floating-point reduction order to excuse a difference.

```bash
V=/usr/local/lib/python3.12/dist-packages/vllm
docker run --rm --gpus all --entrypoint python3 \
  -v $PWD/files/vllm/models/deepseek_v41/nvidia/engram_disk.py:$V/models/deepseek_v41/nvidia/engram_disk.py:ro \
  -v $PWD/tests:/t:ro vllm/vllm-openai:v0.30.0-aarch64 /t/test_engram_disk_equiv.py
```

## Running it

The REAP-256E repository does not ship Engram shards 47 and 48; take them from
the original model (they are byte-identical to the copies other V4.1 recipes
use, so you may already have them locally).

```bash
export WEIGHTS=/path/to/DeepSeek-V4.1-Flash-REAP-256E
export ENGRAM=/path/to/dir/with/shards/47/48
export HEAD_IP=... IB_HCA=... IB_GID=... FABRIC_IF=... ADDR_RANGE=...
export WORK=$HOME/dsv41-work FIPATCH=$WORK/fipatch   # from build-fi.sh

./scripts/serve.sh 1   # worker node
./scripts/serve.sh 0   # head node, serves on :8888
```

The RoCE v2 IPv4 GID index is per node and can move after a reboot, so read it
from `/sys/class/infiniband/<dev>/ports/1/gids` each time instead of pinning a
remembered value.

## Should you use this

Probably not as a service. REAP-256E is a pruned checkpoint (text perplexity
+14.1% by its author's own measurement) capped at 32K context here, and on the
same two boxes an EXL3 recipe such as
[MiaAI-Lab's](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)
decodes at 26 to 37 tok/s. This port is useful as evidence of what the stock
engine still lacks, and as a starting point when 0.30 or later gains the
missing pieces.

## Credits

- [Libertai/dsv41-flash-vllm-2x-spark](https://github.com/Libertai/dsv41-flash-vllm-2x-spark) (MIT):
  the two-node recipe this ports, its launcher shape and its verification script.
- [tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark) (MIT):
  origin of the SM12x patch set, including disk-backed Engram, page geometry and
  the top-k fix, each with its own diff, test and notes.
- [LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E):
  the checkpoint, with its pruning method and quality cost documented.
- [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0): `files/` are
  modified copies of vLLM source.

## License

Apache-2.0, matching vLLM, since `files/` and `patches/` are derived from vLLM
source. See [NOTICE](NOTICE) for the per-source attribution.
