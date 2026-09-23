#!/usr/bin/env bash
# REAP-256E on stock vLLM 0.30.0 across two GB10 (TP=2, RoCE), experiment only.
# Derived from Libertai/dsv41-flash-vllm-2x-spark (MIT); the 09-11 nightly patch
# set is replaced by the 0.30 port in ../files (see ../README.md).
#
# usage: serve.sh <0|1>      rank 0 = head (serves the API), rank 1 = headless worker
#        serve.sh stop       stop this node's container
#
# Required: WEIGHTS, ENGRAM, HEAD_IP, IB_HCA, FABRIC_IF, ADDR_RANGE, IB_GID.
# Read them off your own fabric (ibv_devices, ip -br a, and the RoCE v2 IPv4 GID
# index in /sys/class/infiniband/<dev>/ports/1/gids); the GID index can change
# after a reboot, so check it every time rather than pinning a remembered value.
set -euo pipefail

NODE_RANK="${1:?usage: run-v030.sh <0|1|stop>}"
NAME="dsv41-v030-r${NODE_RANK}"
if [ "$NODE_RANK" = stop ]; then
    for n in dsv41-v030-r0 dsv41-v030-r1; do docker stop -t 90 "$n" 2>/dev/null || true; done
    exit 0
fi

HERE="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.30.0-aarch64}"
WORK="${WORK:-$HOME/dsv41-work}"
FIPATCH="${FIPATCH:-$WORK/fipatch}"
WEIGHTS="${WEIGHTS:?set WEIGHTS to the REAP-256E checkpoint directory}"
ENGRAM="${ENGRAM:?set ENGRAM to the directory holding original shards 47 and 48}"
CKPT="$WORK/reap-ckpt"
HEAD_IP="${HEAD_IP:?head node address on the fast fabric}"
IB_HCA="${IB_HCA:?RoCE device, see ibv_devices}"
IB_GID="${IB_GID:?RoCE v2 IPv4 GID index on both nodes}"
FABRIC_IF="${FABRIC_IF:?fabric interface name}"
ADDR_RANGE="${ADDR_RANGE:?fabric subnet, e.g. 10.0.0.0/24}"
MPORT="${MPORT:-29801}"; PORT="${PORT:-8888}"
GMU="${GMU:-0.90}"; MAXLEN="${MAXLEN:-32768}"; SEQS="${SEQS:-8}"; MAX_BATCHED="${MAX_BATCHED:-4096}"
SPEC="${SPEC:-dspark}"; SPEC_K="${SPEC_K:-5}"
THINKING="${THINKING:-false}"
SITE=/usr/local/lib/python3.12/dist-packages
FI=$SITE/flashinfer
AOT=$SITE/flashinfer_jit_cache/jit_cache/sparse_mla_sm120

case "$NODE_RANK" in 0) HEADLESS="" ;; 1) HEADLESS="--headless" ;; *) echo "rank must be 0 or 1" >&2; exit 2 ;; esac

# Checkpoint view: REAP files from the NAS, Engram shards 47/48 (not in the
# REAP repo) from local NVMe. Both targets are mounted at the same path.
mkdir -p "$CKPT"
for f in "$WEIGHTS"/*.json "$WEIGHTS"/*.safetensors "$WEIGHTS"/encoding "$WEIGHTS"/inference; do
    [ -e "$f" ] && ln -sfn "$f" "$CKPT/$(basename "$f")"
done
for f in model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do
    test -s "$ENGRAM/$f" || { echo "missing $ENGRAM/$f" >&2; exit 3; }
    ln -sfn "$ENGRAM/$f" "$CKPT/$f"
done
n=$(ls "$CKPT"/model-*.safetensors | wc -l)
[ "$n" = 48 ] || { echo "checkpoint has $n/48 shards" >&2; exit 3; }
test -s "$FIPATCH/_sparse_mla_sm120.py" && test -d "$WORK/fi-cache" \
    || { echo "run make-fipatch.sh and build-fi-local.sh first" >&2; exit 3; }

OTHERS=$(docker ps --format '{{.Names}}' | grep -vE "^(${NAME}${ALLOW_CONTAINERS:+|$ALLOW_CONTAINERS})$" || true)
[ -z "$OTHERS" ] || { echo "other containers running: $OTHERS" >&2; exit 4; }

PATCH_MOUNTS=""
for f in $(cd "$HERE/files" && find vllm -name '*.py'); do
    PATCH_MOUNTS="$PATCH_MOUNTS -v $HERE/files/$f:$SITE/$f:ro"
done

# Memory watchdog: headroom after startup is about 2 GiB, and an unlucky prefill
# can take the node down before the OOM killer reacts. The upstream recipe ships
# scripts/mem-watchdog.sh; point WATCHDOG at it (or your own) to arm it.
if [ -n "${WATCHDOG:-}" ] && [ -x "$WATCHDOG" ]; then
    FLOOR_MB="${WATCHDOG_FLOOR_MB:-250}" setsid nohup "$WATCHDOG" "$NAME" \
        >>"${WATCHDOG_LOG:-$WORK/mem-watchdog.log}" 2>&1 < /dev/null &
    echo "memory watchdog armed on $NAME"
else
    echo "WARNING: no memory watchdog (set WATCHDOG=/path/to/mem-watchdog.sh)" >&2
fi

SPEC_ARGS=()
if [ "$SPEC" = dspark ]; then
    SPEC_ARGS=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$SPEC_K,\"draft_sample_method\":\"greedy\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":false}")
fi

exec docker run --rm --name "$NAME" --gpus all \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
  --oom-score-adj 500 \
  -v "$WORK:$WORK" -v "$WEIGHTS:$WEIGHTS:ro" -v "$ENGRAM:$ENGRAM:ro" \
  $PATCH_MOUNTS \
  -v "$FIPATCH/sparse_mla_sm120_decode_dsv4.cu:$FI/data/csrc/sparse_mla_sm120_decode_dsv4.cu:ro" \
  -v "$FIPATCH/sparse_mla_sm120_prefill.cu:$FI/data/csrc/sparse_mla_sm120_prefill.cu:ro" \
  -v "$FIPATCH/_sparse_mla_sm120.py:$FI/mla/_sparse_mla_sm120.py:ro" \
  -v "$WORK/fi-aot-empty:$AOT:ro" \
  -v "$WORK/fi-cache:/root/.cache/flashinfer" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_CACHE_ROOT="$WORK/cache/vllm-v030" -e TILELANG_CACHE_DIR="$WORK/tl-cache" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=5400 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e VLLM_HAS_FLASHINFER_CUBIN=1 -e VLLM_DEEP_GEMM_WARMUP=skip -e PYTHONUNBUFFERED=1 \
  -e DSV41_ENGRAM_DISK=1 -e DSV41_ENGRAM_DISK_THREADS=32 -e DSV41_ENGRAM_DISK_CHUNK=16 \
  -e CUTE_DSL_ARCH=sm_121a -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$IB_HCA" \
  -e NCCL_IB_GID_AUTO=0 -e NCCL_IB_GID_INDEX="$IB_GID" \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE="$ADDR_RANGE" \
  -e NCCL_SOCKET_IFNAME="$FABRIC_IF" -e GLOO_SOCKET_IFNAME="$FABRIC_IF" \
  -e TP_SOCKET_IFNAME="$FABRIC_IF" -e MN_IF_NAME="$FABRIC_IF" \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=1 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  --entrypoint vllm "$IMAGE" serve "$CKPT" \
    --served-model-name dsv41-reap256-v030 \
    --host 0.0.0.0 --port "$PORT" --trust-remote-code \
    --tensor-parallel-size 2 --gpu-memory-utilization "$GMU" --max-model-len "$MAXLEN" \
    --max-num-seqs "$SEQS" --max-num-batched-tokens "$MAX_BATCHED" \
    --enforce-eager --language-model-only \
    --engram-config '{"cpu_offload": false}' \
    --kernel-config '{"enable_flashinfer_autotune": false, "enable_cutedsl_warmup": false, "enable_jit_warmup": true, "sparse_indexer_topk_backend": "per_row"}' \
    --default-chat-template-kwargs "{\"thinking\": $THINKING}" \
    --reasoning-parser deepseek_v41 --enable-auto-tool-choice --tool-call-parser deepseek_v41 \
    "${SPEC_ARGS[@]}" \
    --distributed-executor-backend mp --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" $HEADLESS
