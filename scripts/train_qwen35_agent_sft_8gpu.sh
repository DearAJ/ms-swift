#!/usr/bin/env bash
# Full-parameter Agent SFT. Usage:
#   MODEL=/path/to/Qwen3.5-9B OUTPUT_DIR=output/agent-sft \
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./scripts/train_qwen35_agent_sft_8gpu.sh data/tb2_sft.jsonl
set -euo pipefail

DATASET_SRC="${1:?Usage: MODEL=/path/to/model OUTPUT_DIR=/path/to/output $0 data.jsonl}"
MODEL="${MODEL:?Set MODEL to the local model directory}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the training-output directory}"

# The small set of tunable training defaults. Override any at launch time.
MAX_LENGTH="${MAX_LENGTH:-16384}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-qwen35-agent-sft}"
SWANLAB_EXP_NAME="${SWANLAB_EXP_NAME:-agent-sft-$(date +%Y%m%d-%H%M%S)}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"
export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE="${NPROC_PER_NODE:-$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES}" | wc -l)}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SWIFT_BIN="${SWIFT_BIN:-swift}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET_SRC%.jsonl}_normalized.jsonl"

[[ -f "${DATASET_SRC}" ]] || { echo "Dataset not found: ${DATASET_SRC}" >&2; exit 1; }
[[ -f "${MODEL}/config.json" ]] || { echo "Invalid model directory: ${MODEL}" >&2; exit 1; }

# The agent template requires alternating query/response turns and explicit loss flags.
"${PYTHON_BIN}" "${SCRIPT_DIR}/normalize_agent_dataset.py" "${DATASET_SRC}" "${DATASET}"

SWANLAB_ARGS=(
    --report_to swanlab
    --swanlab_project "${SWANLAB_PROJECT}"
    --swanlab_exp_name "${SWANLAB_EXP_NAME}"
    --swanlab_mode "${SWANLAB_MODE}"
)
if [[ -n "${SWANLAB_API_KEY}" ]]; then
    SWANLAB_ARGS+=(--swanlab_token "${SWANLAB_API_KEY}")
fi

"${SWIFT_BIN}" sft \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --max_length "${MAX_LENGTH}" \
    --gradient_checkpointing true \
    --deepspeed zero3 \
    --split_dataset_ratio 0.01 \
    --save_only_model true \
    "${SWANLAB_ARGS[@]}"
