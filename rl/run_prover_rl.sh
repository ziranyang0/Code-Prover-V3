#!/usr/bin/env bash
# Official-Miles fully-async Code-Prover RL launcher. Ray must already expose
# the complete supported allocation, including the 8x8 and 10x8 H200 campaign topologies.
set -euo pipefail

: "${MILES_ROOT:?set MILES_ROOT to official Miles}"
: "${HF_CHECKPOINT:?set HF_CHECKPOINT}"
: "${MEGATRON_CHECKPOINT:?set MEGATRON_CHECKPOINT}"
: "${RUN_DIR:?set RUN_DIR to a new or resumable output directory}"
: "${E2B_API_KEY_FILE:?set E2B_API_KEY_FILE to a staged 0600 key file}"
: "${CODEPROVER_PATCH_SHA256:?set CODEPROVER_PATCH_SHA256}"
: "${MILES_PATCH_SHA256:?set MILES_PATCH_SHA256}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${CODEPROVER_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
TASK_ROOT="${TASK_ROOT:-${REPO_ROOT}/tasks/trainset_problems_300}"
PROMPT_DATA="${PROMPT_DATA:-${REPO_ROOT}/rl/data/trainset_problems_300.jsonl}"
if [[ -n "${E2B_API_URL:-}${E2B_DOMAIN:-}" ]]; then
  : "${E2B_API_URL:?set E2B_API_URL together with E2B_DOMAIN}"
  : "${E2B_DOMAIN:?set E2B_DOMAIN together with E2B_API_URL}"
  if [[ "${E2B_DOMAIN}" != "e2b.app" ]]; then
    : "${E2B_TEMPLATE_ID:?set a template built on the selected E2B-compatible service}"
  fi
else
  E2B_API_URL="https://api.e2b.app"
  E2B_DOMAIN="e2b.app"
fi
export E2B_API_URL E2B_DOMAIN
: "${E2B_TEMPLATE_ID:?set a template built on the selected E2B-compatible service}"
MILES_SOURCE_COMMIT="${MILES_SOURCE_COMMIT:-12d80fea77d40087724a6910b08a0013a8b34d48}"
CODEPROVER_SOURCE_COMMIT="${CODEPROVER_SOURCE_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}"
MEGATRON_LM_ROOT="${MEGATRON_LM_ROOT:-/root/Megatron-LM}"
NNODES="${NNODES:-${MLP_WORKER_NUM:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${MLP_WORKER_GPU:-8}}"
NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
SAVE_RETAIN_INTERVAL="${SAVE_RETAIN_INTERVAL:-}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-65536}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-32768}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.65}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-16}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-256}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-2}"
PROVER_MAX_TOTAL_TOKENS="${PROVER_MAX_TOTAL_TOKENS:-32768}"
PROVER_MAX_TURNS="${PROVER_MAX_TURNS:-64}"
PROVER_MAX_TOKENS_PER_TURN="${PROVER_MAX_TOKENS_PER_TURN:-2048}"
PROVER_MAX_TRUNCATION_NUDGES="${PROVER_MAX_TRUNCATION_NUDGES:-1}"
PROVER_MAX_TOOL_RESULT_TOKENS="${PROVER_MAX_TOOL_RESULT_TOKENS:-4096}"
PROVER_WALL_TIME_BUDGET_SEC="${PROVER_WALL_TIME_BUDGET_SEC:-1200}"
PROVER_EPISODE_TIMEOUT_SEC="${PROVER_EPISODE_TIMEOUT_SEC:-2400}"
PROVER_ASYNC_NO_PROGRESS_TIMEOUT_SEC="${PROVER_ASYNC_NO_PROGRESS_TIMEOUT_SEC:-$((PROVER_EPISODE_TIMEOUT_SEC + 300))}"
ROUTER_BALANCE_ABS_THRESHOLD="${ROUTER_BALANCE_ABS_THRESHOLD:-1}"
LOAD_DEBUG_ROLLOUT_DATA="${LOAD_DEBUG_ROLLOUT_DATA:-}"
LOAD_CHECKPOINT_DIR="${LOAD_CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}"
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"
NO_LOAD_OPTIM="${NO_LOAD_OPTIM:-0}"
SAVE_OPTIM="${SAVE_OPTIM:-0}"
RESET_ROLLOUT_DATA_STATE="${RESET_ROLLOUT_DATA_STATE:-0}"
CHECK_WEIGHT_UPDATE_EQUAL="${CHECK_WEIGHT_UPDATE_EQUAL:-0}"
CHECK_WEIGHT_UPDATE_SELECTOR="${CHECK_WEIGHT_UPDATE_SELECTOR:-target}"
CHECK_WEIGHT_UPDATE_SKIP_LIST="${CHECK_WEIGHT_UPDATE_SKIP_LIST:-}"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}"
RUN_ID="${RUN_ID:-formal_math-codeprover-async-e2b-${NNODES}n}"
PROVER_EVAL_PROMPT_DATA="${PROVER_EVAL_PROMPT_DATA:-}"
PROVER_TUNING_PROMPT_DATA="${PROVER_TUNING_PROMPT_DATA:-}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-2}"
PROVER_FULL_EVAL_SAMPLES_PER_PROMPT="${PROVER_FULL_EVAL_SAMPLES_PER_PROMPT:-8}"
OBSERVE_TRAINING_ENTROPY="${OBSERVE_TRAINING_ENTROPY:-0}"
RL_ALGORITHM="${RL_ALGORITHM:-grpo}"
ALGORITHM_ARGS=()
case "${RL_ALGORITHM}" in
  grpo) ;;
  dr_grpo)
    ALGORITHM_ARGS=(--disable-grpo-std-normalization
      --policy-loss-normalization-constant "${ROLLOUT_MAX_RESPONSE_LEN}")
    ;;
  *) echo "unsupported RL_ALGORITHM: ${RL_ALGORITHM}" >&2; exit 2 ;;
esac

# Match the D3 deterministic recipe before importing Torch in any process.
# Ray receives these values through RUNTIME_ENV_JSON below.
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
export NCCL_ALGO="${NCCL_ALGO:-^NVLS}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}"

case "${NNODES}:${GPUS_PER_NODE}" in
  4:8)
    # Training-only replay uses the production 32-GPU actor without rollout nodes.
    [[ -n "${LOAD_DEBUG_ROLLOUT_DATA}" ]] || { echo "4x8 requires load_debug_rollout_data" >&2; exit 2; }
    ACTOR_NODES=4
    ACTOR_GPUS_PER_NODE=8
    ACTOR_GPUS=32
    ROLLOUT_GPUS=0
    ROLLOUT_GPUS_PER_ENGINE=1
    SGLANG_EXPERT_PARALLEL=1
    EXPERT_PARALLEL=4
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
    export NCCL_MNNVL_ENABLE=0
    ;;
  1:8)
    ACTOR_NODES=1
    ACTOR_GPUS_PER_NODE=4
    ACTOR_GPUS=4
    ROLLOUT_GPUS=4
    ROLLOUT_GPUS_PER_ENGINE=4
    SGLANG_EXPERT_PARALLEL=4
    EXPERT_PARALLEL=4
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
    ;;
  2:4)
    # Johor GB300 nodes expose four 276-GiB GPUs each.  The full 35B-A3B
    # rollout model fits on one GPU, so use four independent one-GPU engines
    # behind the Miles router instead of one four-GPU EP engine.  The actor
    # remains EP4 on the other node.
    ACTOR_NODES=1
    ACTOR_GPUS_PER_NODE=4
    ACTOR_GPUS=4
    ROLLOUT_GPUS=4
    ROLLOUT_GPUS_PER_ENGINE=1
    SGLANG_EXPERT_PARALLEL=1
    EXPERT_PARALLEL=4
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
    ;;
  4:4)
    # One four-GPU GB300 node runs the EP4 actor.  The remaining three nodes
    # expose twelve independent one-GPU rollout engines (DP12).
    ACTOR_NODES=1
    ACTOR_GPUS_PER_NODE=4
    ACTOR_GPUS=4
    ROLLOUT_GPUS=12
    ROLLOUT_GPUS_PER_ENGINE=1
    SGLANG_EXPERT_PARALLEL=1
    EXPERT_PARALLEL=4
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
    ;;
  6:4)
    # Two four-GPU GB300 nodes run the EP4 actor as DP2.  The remaining four
    # nodes expose sixteen independent one-GPU rollout engines (DP16).
    ACTOR_NODES=2
    ACTOR_GPUS_PER_NODE=4
    ACTOR_GPUS=8
    ROLLOUT_GPUS=16
    ROLLOUT_GPUS_PER_ENGINE=1
    SGLANG_EXPERT_PARALLEL=1
    EXPERT_PARALLEL=4
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
    ;;
  16:4)
    # Eight EP4 actor pods (DP8) and thirty-two one-GPU rollout engines.
    ACTOR_NODES=8
    ACTOR_GPUS_PER_NODE=4
    ACTOR_GPUS=32
    ROLLOUT_GPUS=32
    ROLLOUT_GPUS_PER_ENGINE=1
    SGLANG_EXPERT_PARALLEL=1
    EXPERT_PARALLEL=4
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
    ;;
  8:8|10:8)
    # Keep four H200 actor nodes (EP4); additional nodes serve rollouts.
    # The actor world size stays at 32 so optimizer checkpoints remain compatible.
    ACTOR_NODES=4
    ACTOR_GPUS_PER_NODE=8
    ACTOR_GPUS=32
    ROLLOUT_GPUS=$(( (NNODES - ACTOR_NODES) * GPUS_PER_NODE ))
    ROLLOUT_GPUS_PER_ENGINE=1
    SGLANG_EXPERT_PARALLEL=1
    EXPERT_PARALLEL=4
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
    export NCCL_MNNVL_ENABLE=0
    ;;
  2:8)
    ACTOR_NODES=1
    ACTOR_GPUS_PER_NODE=8
    ACTOR_GPUS=8
    ROLLOUT_GPUS=8
    ROLLOUT_GPUS_PER_ENGINE=8
    SGLANG_EXPERT_PARALLEL=8
    EXPERT_PARALLEL=8
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
    ;;
  *)
    echo "supported async topologies are 1x8, 2x4, 4x4, 6x4, 16x4 (Johor GB300), 8x8/10x8 (SG H200), or 2x8" >&2
    exit 2
    ;;
esac
SGLANG_ATTENTION_ARGS=()
if [[ "${NNODES}:${GPUS_PER_NODE}" != "8:8" && "${NNODES}:${GPUS_PER_NODE}" != "10:8" ]]; then
  SGLANG_ATTENTION_ARGS=(--sglang-attention-backend trtllm_mha)
fi
PROVER_SANDBOX_CONCURRENCY="${PROVER_SANDBOX_CONCURRENCY:-$((ROLLOUT_GPUS * 2))}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
OVER_SAMPLING_MULTIPLIER="${OVER_SAMPLING_MULTIPLIER:-1}"
OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * OVER_SAMPLING_MULTIPLIER))}"
PROVER_ASYNC_QUEUE_SIZE="${PROVER_ASYNC_QUEUE_SIZE:-${OVER_SAMPLING_BATCH_SIZE}}"
PROVER_ASYNC_PAUSE_TIMEOUT_SEC="${PROVER_ASYNC_PAUSE_TIMEOUT_SEC:-300}"
DYNAMIC_SAMPLING_FILTER_PATH="${DYNAMIC_SAMPLING_FILTER_PATH:-}"
if (( SAVE_INTERVAL < 1 )); then
  echo "save_interval must be >= 1" >&2
  exit 2
fi
if (( GLOBAL_BATCH_SIZE != ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT )); then
  echo "global_batch_size must equal rollout_batch_size * n_samples_per_prompt" >&2
  exit 2
fi
if (( OVER_SAMPLING_BATCH_SIZE < ROLLOUT_BATCH_SIZE )); then
  echo "over_sampling_batch_size must be >= rollout_batch_size" >&2
  exit 2
fi
if (( PROVER_ASYNC_QUEUE_SIZE < OVER_SAMPLING_BATCH_SIZE )); then
  echo "prover_async_queue_size must be >= over_sampling_batch_size" >&2
  exit 2
fi
if (( PROVER_ASYNC_PAUSE_TIMEOUT_SEC < 1 )); then
  echo "prover_async_pause_timeout_sec must be >= 1" >&2
  exit 2
fi
if (( PROVER_SANDBOX_CONCURRENCY < 1 )); then
  echo "prover_sandbox_concurrency must be >= 1" >&2
  exit 2
fi
if (( PROVER_MAX_TURNS < 1 )); then
  echo "prover_max_turns must be >= 1" >&2
  exit 2
fi
if (( MAX_SEQ_LEN < PROVER_MAX_TOTAL_TOKENS )); then
  echo "max_seq_len must be >= prover_max_total_tokens" >&2
  exit 2
fi
if (( ROLLOUT_MAX_RESPONSE_LEN < PROVER_MAX_TOTAL_TOKENS )); then
  echo "rollout_max_response_len must be >= prover_max_total_tokens" >&2
  exit 2
fi
if (( CONTEXT_PARALLEL_SIZE < 1 || ACTOR_GPUS % CONTEXT_PARALLEL_SIZE != 0 )); then
  echo "context_parallel_size must be positive and divide actor GPU count" >&2
  exit 2
fi
if (( MAX_TOKENS_PER_GPU * CONTEXT_PARALLEL_SIZE < PROVER_MAX_TOTAL_TOKENS )); then
  echo "max_tokens_per_gpu * context_parallel_size must cover one full prover sequence" >&2
  exit 2
fi
if (( MAX_TOKENS_PER_GPU < 1 )); then
  echo "max_tokens_per_gpu must be >= 1" >&2
  exit 2
fi
if (( LOG_PROBS_CHUNK_SIZE < 1 )); then
  echo "log_probs_chunk_size must be >= 1" >&2
  exit 2
fi
if (( SGLANG_SERVER_CONCURRENCY < 1 || SGLANG_SERVER_CONCURRENCY > 16 )); then
  echo "sglang_server_concurrency must be in [1, 16]" >&2
  exit 2
fi
if [[ "${NO_LOAD_OPTIM}" != "0" && "${NO_LOAD_OPTIM}" != "1" ]]; then
  echo "no_load_optim must be 0 or 1" >&2
  exit 2
fi
if [[ "${SAVE_OPTIM}" != "0" && "${SAVE_OPTIM}" != "1" ]]; then
  echo "save_optim must be 0 or 1" >&2
  exit 2
fi
if [[ "${RESET_ROLLOUT_DATA_STATE}" != "0" && "${RESET_ROLLOUT_DATA_STATE}" != "1" ]]; then
  echo "reset_rollout_data_state must be 0 or 1" >&2
  exit 2
fi
if [[ "${CHECK_WEIGHT_UPDATE_EQUAL}" != "0" && "${CHECK_WEIGHT_UPDATE_EQUAL}" != "1" ]]; then
  echo "check_weight_update_equal must be 0 or 1" >&2
  exit 2
fi
if [[ "${CHECK_WEIGHT_UPDATE_SELECTOR}" != "all" && "${CHECK_WEIGHT_UPDATE_SELECTOR}" != "target" && "${CHECK_WEIGHT_UPDATE_SELECTOR}" != "draft" ]]; then
  echo "check_weight_update_selector must be all, target, or draft" >&2
  exit 2
fi
python3 - "${OPTIMIZER_OFFLOAD_FRACTION}" <<'PY'
import sys

fraction = float(sys.argv[1])
if not 0.0 <= fraction <= 1.0:
    raise SystemExit("optimizer_offload_fraction must be in [0, 1]")
PY
OPTIMIZER_CPU_OFFLOAD_ENABLED="$(python3 - "${OPTIMIZER_OFFLOAD_FRACTION}" <<'PY'
import sys

print(int(float(sys.argv[1]) > 0.0))
PY
)"
if (( PROVER_WALL_TIME_BUDGET_SEC < 1 )); then
  echo "prover_wall_time_budget_sec must be >= 1" >&2
  exit 2
fi
if (( PROVER_EPISODE_TIMEOUT_SEC <= PROVER_WALL_TIME_BUDGET_SEC )); then
  echo "prover_episode_timeout_sec must exceed prover_wall_time_budget_sec" >&2
  exit 2
fi
if (( GLOBAL_BATCH_SIZE % ACTOR_GPUS != 0 )); then
  echo "rollout_batch_size * n_samples must be divisible by actor GPU count" >&2
  exit 2
fi
DYNAMIC_SAMPLING_ARGS=(--over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}")
if [[ -n "${DYNAMIC_SAMPLING_FILTER_PATH}" ]]; then
  DYNAMIC_SAMPLING_ARGS+=(--dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}")
fi
DEBUG_ROLLOUT_ARGS=()
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA}" ]]; then
  if [[ "${LOAD_DEBUG_ROLLOUT_DATA}" != *"{rollout_id}"* && ! -r "${LOAD_DEBUG_ROLLOUT_DATA}" ]]; then
    echo "load_debug_rollout_data is not readable: ${LOAD_DEBUG_ROLLOUT_DATA}" >&2
    exit 2
  fi
  DEBUG_ROLLOUT_ARGS+=(--load-debug-rollout-data "${LOAD_DEBUG_ROLLOUT_DATA}")
fi
LOAD_OPTIM_ARGS=()
if [[ "${NO_LOAD_OPTIM}" == "1" ]]; then
  LOAD_OPTIM_ARGS+=(--no-load-optim)
fi
SAVE_OPTIM_ARGS=()
if [[ "${SAVE_OPTIM}" == "0" ]]; then
  SAVE_OPTIM_ARGS+=(--no-save-optim)
fi
RESET_DATA_ARGS=()
if [[ "${RESET_ROLLOUT_DATA_STATE}" == "1" ]]; then
  RESET_DATA_ARGS+=(--prover-reset-rollout-data-state)
fi
SAVE_RETENTION_ARGS=()
if [[ -n "${SAVE_RETAIN_INTERVAL}" ]]; then
  if (( SAVE_RETAIN_INTERVAL < 1 || SAVE_RETAIN_INTERVAL % SAVE_INTERVAL != 0 )); then
    echo "save_retain_interval must be positive and divisible by save_interval" >&2
    exit 2
  fi
  SAVE_RETENTION_ARGS+=(--save-retain-interval "${SAVE_RETAIN_INTERVAL}")
fi
OPTIMIZER_OFFLOAD_ARGS=()
if [[ "${OPTIMIZER_CPU_OFFLOAD_ENABLED}" == "1" ]]; then
  OPTIMIZER_OFFLOAD_ARGS+=(
    --optimizer-cpu-offload
    --optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION}"
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
  )
fi
WEIGHT_CHECK_ARGS=()
if [[ "${CHECK_WEIGHT_UPDATE_EQUAL}" == "1" ]]; then
  WEIGHT_CHECK_ARGS+=(
    --check-weight-update-equal
    --check-weight-update-selector "${CHECK_WEIGHT_UPDATE_SELECTOR}"
  )
  if [[ -n "${CHECK_WEIGHT_UPDATE_SKIP_LIST}" ]]; then
    read -r -a WEIGHT_CHECK_SKIP_ITEMS <<<"${CHECK_WEIGHT_UPDATE_SKIP_LIST}"
    WEIGHT_CHECK_ARGS+=(--check-weight-update-skip-list "${WEIGHT_CHECK_SKIP_ITEMS[@]}")
  fi
fi

export PYTHONPATH="${REPO_ROOT}:${MILES_ROOT}:${MEGATRON_LM_ROOT}:${MEGATRON_LM_ROOT}/submodules/apodex-core${PYTHONPATH:+:${PYTHONPATH}}"
EVAL_ARGS=()
if [[ -n "${PROVER_EVAL_PROMPT_DATA}" ]]; then
  EVAL_ARGS=(--eval-function-path rl.evaluate.generate_rollout_eval
    --eval-interval "${EVAL_INTERVAL}"
    --eval-prompt-data d3_hard "${PROVER_EVAL_PROMPT_DATA}"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
    --prover-full-eval-samples-per-prompt "${PROVER_FULL_EVAL_SAMPLES_PER_PROMPT}"
    --prover-eval-prompt-data "${PROVER_EVAL_PROMPT_DATA}")
  python3 - "${PROMPT_DATA}" "${PROVER_EVAL_PROMPT_DATA}" "${PROVER_TUNING_PROMPT_DATA}" <<'PY_EVAL'
import sys
from rl.make_eval_data import assert_disjoint
assert_disjoint(sys.argv[1], [path for path in sys.argv[2:] if path])
PY_EVAL
fi
ENTROPY_ARGS=()
if [[ "${OBSERVE_TRAINING_ENTROPY}" == "1" ]]; then
  ENTROPY_ARGS=(--observe-training-entropy)
fi
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF
export RUN_ID RUN_DIR E2B_API_KEY_FILE CODEPROVER_SOURCE_COMMIT MILES_SOURCE_COMMIT
export CODEPROVER_PATCH_SHA256 MILES_PATCH_SHA256
export LOAD_CHECKPOINT_DIR OPTIMIZER_OFFLOAD_FRACTION NO_LOAD_OPTIM SAVE_OPTIM
export RESET_ROLLOUT_DATA_STATE
mkdir -p "${RUN_DIR}" "${RUN_DIR}/checkpoints" "${RUN_DIR}/dump_details"

PROVER_JUDGE_ARGS=(
  --prover-judge-mode "${PROVER_JUDGE_MODE:-comparator}"
  --prover-comparator-image "${PROVER_COMPARATOR_IMAGE:-}"
  --prover-comparator-queue-dir "${PROVER_COMPARATOR_QUEUE_DIR:-}"
  --prover-comparator-timeout-sec "${PROVER_COMPARATOR_TIMEOUT_SEC:-1200}"
  --prover-comparator-concurrency "${PROVER_COMPARATOR_CONCURRENCY:-2}"
  --prover-proof-artifacts-dir "${PROVER_PROOF_ARTIFACTS_DIR:-${RUN_DIR}/proofs}"
)
PREFLIGHT_MODE_ARGS=()
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA}" ]]; then
  PREFLIGHT_MODE_ARGS+=(--training-only)
fi
PREFLIGHT_LOAD_ARGS=(
  --load-checkpoint-dir "${LOAD_CHECKPOINT_DIR}"
  --optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION}"
)
if [[ "${NO_LOAD_OPTIM}" == "1" ]]; then
  PREFLIGHT_LOAD_ARGS+=(--no-load-optim)
fi

python3 "${REPO_ROOT}/rl/preflight.py" \
  --repo-root "${REPO_ROOT}" \
  --miles-root "${MILES_ROOT}" \
  --megatron-lm-root "${MEGATRON_LM_ROOT}" \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --megatron-checkpoint "${MEGATRON_CHECKPOINT}" \
  --prompt-data "${PROMPT_DATA}" \
  --task-root "${TASK_ROOT}" \
  --dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}" \
  --nodes "${NNODES}" \
  --gpus-per-node "${GPUS_PER_NODE}" \
  --actor-gpus "${ACTOR_GPUS}" \
  --rollout-gpus "${ROLLOUT_GPUS}" \
  --actor-tensor-parallel-size 1 \
  --actor-pipeline-parallel-size 1 \
  --actor-context-parallel-size "${CONTEXT_PARALLEL_SIZE}" \
  --actor-expert-parallel-size "${EXPERT_PARALLEL}" \
  "${PREFLIGHT_LOAD_ARGS[@]}" \
  "${PREFLIGHT_MODE_ARGS[@]}" \
  "${PROVER_JUDGE_ARGS[@]}" \
  --e2b-template "${E2B_TEMPLATE_ID}" \
  --expected-repo-commit "${CODEPROVER_SOURCE_COMMIT}" \
  --expected-repo-patch-sha256 "${CODEPROVER_PATCH_SHA256}" \
  --expected-miles-commit "${MILES_SOURCE_COMMIT}" \
  --expected-miles-patch-sha256 "${MILES_PATCH_SHA256}" \
  --allow-dirty-source \
  >"${RUN_DIR}/preflight.json"

# This allocates and destroys a real sandbox before any model consumes GPUs.
# A training-only replay never instantiates rollout engines or sandboxes.
if [[ -z "${LOAD_DEBUG_ROLLOUT_DATA}" ]]; then
  python3 "${REPO_ROOT}/rl/e2b_smoke.py" --template "${E2B_TEMPLATE_ID}" \
    >"${RUN_DIR}/e2b-preflight.json"
else
  printf '{"skipped":true,"reason":"load_debug_rollout_data"}\n' \
    >"${RUN_DIR}/e2b-preflight.json"
fi

source "${MILES_ROOT}/scripts/models/qwen3.6-35B-A3B.sh"
MODEL_ARGS_NO_MTP=()
skip_next=0
for arg in "${MODEL_ARGS[@]}"; do
  if (( skip_next )); then
    skip_next=0
    continue
  fi
  if [[ "${arg}" == "--mtp-num-layers" ]]; then
    skip_next=1
    continue
  fi
  MODEL_ARGS_NO_MTP+=("${arg}")
done

RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json
import os

names = {
    "PYTHONPATH", "PYTHONUNBUFFERED", "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR",
    "RUN_ID", "RUN_DIR", "ATTEMPT_ID", "E2B_WARM_POOL_SIZE", "E2B_WARM_POOL_WAIT_TIMEOUT_SEC", "E2B_API_KEY_FILE", "E2B_API_URL", "E2B_DOMAIN", "E2B_CREATE_CONCURRENCY",
    "E2B_SANDBOX_PROVIDER", "E2B_VALIDATE_API_KEY", "SSL_CERT_FILE", "NO_PROXY", "no_proxy",
    "E2B_CREATE_METADATA_JSON", "E2B_CREATE_TIMEOUT_SEC", "E2B_BOOTSTRAP_PROFILE",
    "CODEPROVER_SOURCE_COMMIT", "CODEPROVER_PATCH_SHA256",
    "MILES_SOURCE_COMMIT", "MILES_PATCH_SHA256", "PYTORCH_CUDA_ALLOC_CONF",
    "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_ALGO", "NCCL_MNNVL_ENABLE",
    "NCCL_CUMEM_ENABLE", "CUBLAS_WORKSPACE_CONFIG", "TORCHDYNAMO_DISABLE",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
}
print(json.dumps({"env_vars": {name: os.environ[name] for name in names if name in os.environ}}))
PY
)"

cat >"${RUN_DIR}/run-manifest.txt" <<EOF
run_id=${RUN_ID}
rl_algorithm=${RL_ALGORITHM}
codeprover_commit=${CODEPROVER_SOURCE_COMMIT}
codeprover_patch_sha256=${CODEPROVER_PATCH_SHA256}
miles_commit=${MILES_SOURCE_COMMIT}
miles_patch_sha256=${MILES_PATCH_SHA256}
nodes=${NNODES}
actor_nodes=${ACTOR_NODES}
actor_gpus_per_node=${ACTOR_GPUS_PER_NODE}
actor_gpus=${ACTOR_GPUS}
rollout_gpus=${ROLLOUT_GPUS}
rollout_gpus_per_engine=${ROLLOUT_GPUS_PER_ENGINE}
sglang_expert_parallel=${SGLANG_EXPERT_PARALLEL}
rollout_batch_size=${ROLLOUT_BATCH_SIZE}
n_samples_per_prompt=${N_SAMPLES_PER_PROMPT}
global_batch_size=${GLOBAL_BATCH_SIZE}
over_sampling_batch_size=${OVER_SAMPLING_BATCH_SIZE}
dynamic_sampling_filter_path=${DYNAMIC_SAMPLING_FILTER_PATH:-disabled}
prover_eval_prompt_data=${PROVER_EVAL_PROMPT_DATA:-disabled}
eval_interval=${EVAL_INTERVAL}
n_samples_per_eval_prompt=${N_SAMPLES_PER_EVAL_PROMPT}
prover_full_eval_samples_per_prompt=${PROVER_FULL_EVAL_SAMPLES_PER_PROMPT}
observe_training_entropy=${OBSERVE_TRAINING_ENTROPY}
prover_max_total_tokens=${PROVER_MAX_TOTAL_TOKENS}
prover_max_turns=${PROVER_MAX_TURNS}
prover_max_tokens_per_turn=${PROVER_MAX_TOKENS_PER_TURN}
prover_max_truncation_nudges=${PROVER_MAX_TRUNCATION_NUDGES}
prover_max_tool_result_tokens=${PROVER_MAX_TOOL_RESULT_TOKENS}
prover_wall_time_budget_sec=${PROVER_WALL_TIME_BUDGET_SEC}
prover_episode_timeout_sec=${PROVER_EPISODE_TIMEOUT_SEC}
prover_sandbox_concurrency=${PROVER_SANDBOX_CONCURRENCY}
max_seq_len=${MAX_SEQ_LEN}
rollout_max_response_len=${ROLLOUT_MAX_RESPONSE_LEN}
max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}
context_parallel_size=${CONTEXT_PARALLEL_SIZE}
sglang_mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}
sglang_max_running_requests=${SGLANG_MAX_RUNNING_REQUESTS}
log_probs_chunk_size=${LOG_PROBS_CHUNK_SIZE}
pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF}
load_debug_rollout_data=${LOAD_DEBUG_ROLLOUT_DATA:-disabled}
load_checkpoint_dir=${LOAD_CHECKPOINT_DIR}
optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION}
optimizer_cpu_offload_enabled=${OPTIMIZER_CPU_OFFLOAD_ENABLED}
no_load_optim=${NO_LOAD_OPTIM}
save_optim=${SAVE_OPTIM}
save_retain_interval=${SAVE_RETAIN_INTERVAL:-disabled}
reset_rollout_data_state=${RESET_ROLLOUT_DATA_STATE}
check_weight_update_equal=${CHECK_WEIGHT_UPDATE_EQUAL}
check_weight_update_selector=${CHECK_WEIGHT_UPDATE_SELECTOR}
check_weight_update_skip_list=${CHECK_WEIGHT_UPDATE_SKIP_LIST:-disabled}
sglang_server_concurrency=${SGLANG_SERVER_CONCURRENCY}
prover_async_queue_size=${PROVER_ASYNC_QUEUE_SIZE}
prover_async_pause_timeout_sec=${PROVER_ASYNC_PAUSE_TIMEOUT_SEC}
prover_async_max_group_retries=${PROVER_ASYNC_MAX_GROUP_RETRIES:-2}
prover_async_max_consecutive_failures=${PROVER_ASYNC_MAX_CONSECUTIVE_FAILURES:-0}
prover_async_no_progress_timeout_sec=${PROVER_ASYNC_NO_PROGRESS_TIMEOUT_SEC}
router_balance_abs_threshold=${ROUTER_BALANCE_ABS_THRESHOLD}
sandbox=e2b
e2b_api_url=${E2B_API_URL}
e2b_domain=${E2B_DOMAIN}
e2b_sandbox_provider=${E2B_SANDBOX_PROVIDER:-default}
e2b_validate_api_key=${E2B_VALIDATE_API_KEY:-true}
e2b_ca_bundle=${SSL_CERT_FILE:-system}
e2b_bootstrap_profile=${E2B_BOOTSTRAP_PROFILE:-prover}
e2b_create_timeout_sec=${E2B_CREATE_TIMEOUT_SEC:-120}
e2b_create_concurrency=${E2B_CREATE_CONCURRENCY:-8}
e2b_warm_pool_size=${E2B_WARM_POOL_SIZE:-0}
e2b_warm_pool_wait_timeout_sec=${E2B_WARM_POOL_WAIT_TIMEOUT_SEC:-1200}
e2b_create_metadata_json=${E2B_CREATE_METADATA_JSON:-}
e2b_template=${E2B_TEMPLATE_ID}
prompt_data=${PROMPT_DATA}
task_root=${TASK_ROOT}
hf_checkpoint=${HF_CHECKPOINT}
megatron_checkpoint=${MEGATRON_CHECKPOINT}
EOF

ray job submit \
  --address="${RAY_DASHBOARD_ADDRESS}" \
  --submission-id="${RUN_ID}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- \
  python3 "${MILES_ROOT}/train_async.py" \
    --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES:-60}" \
    --deterministic-mode \
    --start-rollout-id "${START_ROLLOUT_ID:-0}" \
    --actor-num-nodes "${ACTOR_NODES}" \
    --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
    --num-gpus-per-node "${GPUS_PER_NODE}" \
    --rollout-num-gpus "${ROLLOUT_GPUS}" \
    "${MODEL_ARGS_NO_MTP[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --ref-load "${MEGATRON_CHECKPOINT}" \
    --load "${LOAD_CHECKPOINT_DIR}" \
    "${LOAD_OPTIM_ARGS[@]}" \
    "${WEIGHT_CHECK_ARGS[@]}" \
    --save "${RUN_DIR}/checkpoints" \
    --save-interval "${SAVE_INTERVAL}" \
    "${SAVE_RETENTION_ARGS[@]}" \
    "${SAVE_OPTIM_ARGS[@]}" \
    --prompt-data "${PROMPT_DATA}" \
    --input-key prompt \
    --metadata-key metadata \
    --apply-chat-template \
    --data-source-path rl.persistent_data_source.PersistentRolloutDataSource \
    --rollout-function-path rl.fully_async_rollout.generate_rollout_fully_async \
    "${EVAL_ARGS[@]}" \
    "${ENTROPY_ARGS[@]}" \
    --custom-generate-function-path rl.generate_with_prover.generate \
    --prover-model-path "${HF_CHECKPOINT}" \
    --prover-task-root "${TASK_ROOT}" \
    "${PROVER_JUDGE_ARGS[@]}" \
    --prover-sandbox-backend e2b \
    --prover-e2b-template "${E2B_TEMPLATE_ID}" \
    --prover-sandbox-concurrency "${PROVER_SANDBOX_CONCURRENCY}" \
    --prover-max-turns "${PROVER_MAX_TURNS}" \
    --prover-max-total-tokens "${PROVER_MAX_TOTAL_TOKENS}" \
    --prover-max-tokens-per-turn "${PROVER_MAX_TOKENS_PER_TURN}" \
    --prover-max-truncation-nudges "${PROVER_MAX_TRUNCATION_NUDGES}" \
    --prover-max-tool-result-tokens "${PROVER_MAX_TOOL_RESULT_TOKENS}" \
    --prover-wall-time-budget-sec "${PROVER_WALL_TIME_BUDGET_SEC}" \
    --prover-episode-timeout-sec "${PROVER_EPISODE_TIMEOUT_SEC}" \
    --prover-router-timeout-sec "${PROVER_ROUTER_TIMEOUT_SEC:-180}" \
    --prover-async-queue-size "${PROVER_ASYNC_QUEUE_SIZE}" \
    --prover-async-pause-timeout-sec "${PROVER_ASYNC_PAUSE_TIMEOUT_SEC}" \
    --prover-async-max-consecutive-failures "${PROVER_ASYNC_MAX_CONSECUTIVE_FAILURES:-0}" \
    --prover-async-max-group-retries "${PROVER_ASYNC_MAX_GROUP_RETRIES:-2}" \
    --prover-async-no-progress-timeout-sec "${PROVER_ASYNC_NO_PROGRESS_TIMEOUT_SEC}" \
    "${RESET_DATA_ARGS[@]}" \
    "${DEBUG_ROLLOUT_ARGS[@]}" \
    --num-rollout "${NUM_ROLLOUT}" \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
    "${DYNAMIC_SAMPLING_ARGS[@]}" \
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}" \
    --rollout-max-context-len "${MAX_SEQ_LEN}" \
    --rollout-temperature 0.8 \
    --max-weight-staleness "${MAX_WEIGHT_STALENESS:-2}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --num-steps-per-rollout 1 \
    --seq-length "${MAX_SEQ_LEN}" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size "${CONTEXT_PARALLEL_SIZE}" \
    --expert-model-parallel-size "${EXPERT_PARALLEL}" \
    --expert-tensor-parallel-size 1 \
    --sequence-parallel \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
    --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}" \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --override-opt-param-scheduler \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    "${OPTIMIZER_OFFLOAD_ARGS[@]}" \
    --advantage-estimator grpo \
    "${ALGORITHM_ARGS[@]}" \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    --entropy-coef 0.0 \
    --use-tis \
    --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}" \
    --sglang-ep-size "${SGLANG_EXPERT_PARALLEL}" \
    --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}" \
    --router-balance-abs-threshold "${ROUTER_BALANCE_ABS_THRESHOLD}" \
    --sglang-moe-runner-backend flashinfer_cutlass \
    "${SGLANG_ATTENTION_ARGS[@]}" \
    --sglang-cuda-graph-backend-decode full \
    --sglang-cuda-graph-backend-prefill disabled \
    --sglang-cuda-graph-max-bs-decode 16 \
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}" \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend flash \
    --moe-token-dispatcher-type flex \
    --dump-details "${RUN_DIR}/dump_details" \
    --ci-save-model-hash \
    --log-passrate \
    --skip-eval-before-train
