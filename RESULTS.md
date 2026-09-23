# Measurements, 2026-09-23

Numbers without their conditions are not worth much, so here is everything
needed to judge or repeat these.

## Setup

| | |
|---|---|
| Nodes | 2x GB10, 128 GB unified memory each, direct 200 Gb-class link, NCCL over RoCE v2 |
| Engine | `vllm/vllm-openai:v0.30.0-aarch64`, torch 2.13.0+cu130, FlashInfer 0.6.18.post1 |
| Driver | 580.178.04. The upstream recipe's baseline is 580.173.02, so cross-recipe comparison has this uncontrolled variable |
| Checkpoint | LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E; Engram shards 47/48 from the original model, on local NVMe |
| Weights storage | checkpoint on NFS, Engram on local NVMe |
| Serve flags | TP=2, 32K context, `max-num-seqs 8`, `max-num-batched-tokens 4096`, `gpu-memory-utilization 0.90`, `--enforce-eager`, DSpark k=5, `--language-model-only`, thinking off by default |

## Startup

| Stage | Time |
|---|---|
| 48 main shards over NFS | 22.6 min |
| DSpark drafter pass over the same shards | about 44 min |
| `Model loading took` | 100.04 GiB per rank, 4010 s |
| profile + KV + warmup | 93.8 s |
| Total to serving | about 69 min |

KV pool 372,529 tokens, 11.37x concurrency at 32K per request. After startup
each node had about 2 GiB MemAvailable and roughly 5 GiB of swap in use, but
swap-in during decode was effectively zero. No Xid errors on either node.

## Correctness

Run with the upstream recipe's `verify_serving.py`, unmodified. It deliberately
includes a default-shape request, because probes that always pass optional flags
never test the default behaviour.

- Finite logprobs. This is the NaN detector: NaN output still looks like normal
  text in `content` and shows up only here.
- Known answer 17x19 = 323.
- Tool call emitted and the result round-tripped.
- Needle retrieval: 13,835 tokens (14.0 s) and 26,030 tokens (25.6 s), exact.
- Traditional Chinese: short prompt and an 11,126-token prompt at temperature 0
  and 1.0, zero Simplified characters, coherent.
- Reasoning on: 1022 characters of reasoning, correct answer.
- Reasoning off: a sequence question answered wrong (1024 instead of 110). That
  is the pruned model without reasoning, not a deployment fault.

## Performance

| Measurement | Result |
|---|---|
| Single stream, 400-token Chinese prose, temperature 0 | 7.4 to 15.8 tok/s across runs; median about 9 to 10 on a clean re-run |
| Aggregate at 2 / 4 / 8 concurrent | 12.1 / 22.5 / 31.5 tok/s |
| TTFT cold / warm, 7,813-token prompt | 11.7 s / 0.92 s (prefix cache hits) |
| TTFT under concurrency | mostly 0.7 to 0.9 s, but queued requests waited 20 to 50 s |
| GPU utilization while decoding | 83% to 96%, compute bound |

Method: streaming chat completions, temperature 0, concurrency 1/2/4/8 with two
rounds each; single-stream re-measured separately three times because the
concurrency sweep leaves queueing effects behind. Cold and warm TTFT are
reported separately because the prefix cache hits on repeat, and a warm number
must never be quoted as a cold one.

## Limits and what was not done

- The single-stream spread (7.4 to 15.8) is wide and the sample is small. Treat
  9 to 10 as an order of magnitude, not a precise figure.
- Only one configuration was run. Memory utilization and KV size were never
  tuned, and headroom after startup was only about 2 GiB. That may well explain
  why this is roughly half of the recipe author's published figure, but it was
  not verified, so it is not stated as a conclusion.
- Perplexity was not measured here; the +14.1% text figure is the checkpoint
  author's.
- One or two runs per test, no significance testing.
- Lab conditions: no other workload on the machines during measurement.
