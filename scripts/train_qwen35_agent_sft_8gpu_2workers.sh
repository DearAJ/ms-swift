#!/usr/bin/env bash
# Full-parameter Agent SFT of Qwen3.5-27B on 2 nodes x 8 H100s (16 GPUs).
# Same five-dimension control panel as train_qwen35_agent_sft_8gpu.sh; the only
# multi-node additions are the platform env mapping and NCCL settings below.
#
# The platform injects MASTER_ADDR / MASTER_PORT / NPROC_PER_NODE / WORLD_SIZE
# (= node count) / RANK (= this node's index, 0 for Master) into every pod, and
# runs this same script on both pods. swift's torchrun wrapper reads NNODES and
# NODE_RANK, hence the two-line mapping.
#
# Usage:
#   ./scripts/train_qwen35_agent_sft_8gpu_2workers.sh \
#     --model-input-dir /path/to/Qwen3.5-27B \
#     --model-output-dir output/rsi-qwen35-27b/v1 \
#     --data input/tb4_sft.jsonl
# MODEL and OUTPUT_DIR remain supported for backwards compatibility.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"   # ms-swift root

DATASET_SRC="${PROJECT_DIR}/input/tb4_sft.jsonl"
MODEL_INPUT_DIR="${MODEL_INPUT_DIR:-${MODEL:-}}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-${OUTPUT_DIR:-${PROJECT_DIR}/output/rsi-qwen35-27b/v1}}"
while (( $# > 0 )); do
    case "$1" in
        --data)
            (( $# >= 2 )) || { echo "--data requires a JSONL path" >&2; exit 2; }
            DATASET_SRC="$2"
            shift 2
            ;;
        --data=*)
            DATASET_SRC="${1#--data=}"
            shift
            ;;
        --model-input-dir)
            (( $# >= 2 )) || { echo "--model-input-dir requires a directory" >&2; exit 2; }
            MODEL_INPUT_DIR="$2"
            shift 2
            ;;
        --model-input-dir=*)
            MODEL_INPUT_DIR="${1#--model-input-dir=}"
            shift
            ;;
        --model-output-dir)
            (( $# >= 2 )) || { echo "--model-output-dir requires a directory" >&2; exit 2; }
            MODEL_OUTPUT_DIR="$2"
            shift 2
            ;;
        --model-output-dir=*)
            MODEL_OUTPUT_DIR="${1#--model-output-dir=}"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--data PATH | PATH] --model-input-dir DIR --model-output-dir DIR"
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
        *)
            DATASET_SRC="$1"
            shift
            (( $# == 0 )) || { echo "Only one dataset path may be specified" >&2; exit 2; }
            ;;
    esac
done
MODEL="${MODEL_INPUT_DIR:?Pass --model-input-dir DIR or set MODEL_INPUT_DIR/MODEL}"
OUTPUT_DIR="${MODEL_OUTPUT_DIR}"

# The small set of tunable training defaults. Override any at launch time.
MAX_LENGTH="${MAX_LENGTH:-16384}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-qwen35-27b-agent-sft}"
SWANLAB_EXP_NAME="${SWANLAB_EXP_NAME:-agent-sft-$(date +%Y%m%d-%H%M%S)}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
SWANLAB_API_KEY="${SWANLAB_API_KEY:-XLi4DoFQuZtgS8WBJWFMM}"
export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE="${NPROC_PER_NODE:-$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES}" | wc -l)}"
# Multi-node: map the platform's WORLD_SIZE/RANK onto swift's NNODES/NODE_RANK.
# MASTER_ADDR, MASTER_PORT, and NPROC_PER_NODE are picked up by swift directly.
export NNODES="${NNODES:-${WORLD_SIZE:-1}}"
export NODE_RANK="${NODE_RANK:-${RANK:-0}}"
# NCCL settings recommended by the platform docs for multi-node RoCE clusters.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"   # Unneeded for NCCL >= 2.21
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

SWIFT_BIN="${SWIFT_BIN:-swift}"

# RSI_RECIPE_BEGIN
# Curriculum interface (see docs/source/Instruction/Curriculum.en.md).
# To enable it, run the script for easy, medium, and hard datasets in order,
# changing DATASET_SRC and OUTPUT_DIR at each stage.
CURRICULUM_STAGES="${CURRICULUM_STAGES:-}"    # Comma-separated easy,medium,hard datasets; trained sequentially

# Add entries here to mix multiple datasets (used with --interleave_prob below).
DATASETS=("${DATASET_SRC}")
# DATASETS+=("data/tb4_math.jsonl#2000")                    # Local file, first 2,000 rows
# DATASETS+=("hf::meta-llama/Llama-3.2-1B-Instruct#1000")  # Hugging Face dataset, first 1,000 rows

for ds in "${DATASETS[@]}"; do
    [[ "${ds}" == *"::"* ]] && continue                 # Skip local checks for repository datasets
    ds_path="${ds%%#*}"                                  # Remove the optional #count suffix
    [[ -f "${ds_path}" ]] || { echo "Dataset not found: ${ds_path}" >&2; exit 1; }
done
[[ -f "${MODEL}/config.json" ]] || { echo "Invalid model directory: ${MODEL}" >&2; exit 1; }

SWANLAB_ARGS=(
    --report_to swanlab
    --swanlab_project "${SWANLAB_PROJECT}"
    --swanlab_exp_name "${SWANLAB_EXP_NAME}"
    --swanlab_mode "${SWANLAB_MODE}"
)
if [[ -n "${SWANLAB_API_KEY}" ]]; then
    SWANLAB_ARGS+=(--swanlab_token "${SWANLAB_API_KEY}")
fi

# ============================================================
# Dimension 1: data content, quality, filtering, and cleaning
# ============================================================
# RSI_DATA_BEGIN
DATA_ARGS=(
    --dataset "${DATASETS[@]}"          # Sources: JSONL, CSV, directory, or Hugging Face dataset
    --strict false                      # false drops invalid rows with bounded logging; true exits on error
    --truncation_strategy left          # Overlength handling: left keeps the most recent turns (agent-friendly);
    # --truncation_strategy delete      #   delete drops overlength samples but can crash training when
    #                                    #   many long trajectories meet in the lazy-tokenize fallback loop
    # --columns '{"text1": "query", "text2": "response"}'  # Map source columns to the standard schema
    # --custom_dataset_info data/dataset_info.json          # Register a custom dataset schema
    # --cached_dataset true              # Consume a cache produced by `swift export --to_cached_dataset`
)
# RSI_DATA_END
# For offline row-level cleaning, implement a Python RowPreprocessor and generate
# a clean JSONL before training. See docs/source/Instruction/Custom-data-cleaning.md.
# Curriculum has no single built-in switch. Common approaches are:
#   1. Sort data by difficulty and set --dataset_shuffle false.
#   2. Use a custom callback to change data between stages.
#   3. Run multiple stages and resume each stage from the prior checkpoint.
# See docs/source/Instruction/Curriculum.en.md for deterministic difficulty buckets.

# ============================================================
# Dimension 2: data mixture, sampling, order, curriculum, and packing
# ============================================================
MIX_ARGS=(
    --split_dataset_ratio 0.01          # Randomly reserve 1% for validation
    --data_seed 42                      # Fixed shuffle seed for reproducibility
    --dataset_shuffle true              # Set false to preserve curriculum ordering
    # --interleave_prob 0.3 0.7         # One mixing probability per dataset
    # --stopping_strategy first_exhausted  # first_exhausted or all_exhausted
    # --shuffle_buffer_size 1000        # Shuffle window in streaming mode
    # --group_by_length true            # Group similar lengths to reduce padding
    # --packing true --packing_length 16384  # Pack multiple samples into one sequence
    # --padding_free true               # Remove padding when supported by the template
    # --streaming true                  # Streaming requires --max_steps
)
# RSI_RECIPE_END

# ============================================================
# Dimension 3: loss, masking, supervision scope, and weighting
# ============================================================
# RSI_OBJECTIVE_BEGIN
LOSS_ARGS=(
    --loss_type cross_entropy           # Other built-ins include cosine_similarity, contrastive, and infonce
    --loss_scale default                # default supervises assistant turns; last_round supervises only the final turn
    # --loss_scale last_round+hermes    # Composable weighting: react, qwen, agentflan, ignore_empty_think
    # --mrl_dims '{"32": 1.0, "64": 1.0}'  # Weighted multi-resolution embedding loss
)
# RSI_OBJECTIVE_END
# Per-message supervision can be controlled with a "loss": 0/1 field in each message.

# ============================================================
# Dimension 4: optimizer, learning rate, scheduler, PEFT, and update scope
# ============================================================
# RSI_UPDATE_BEGIN
OPT_ARGS=(
    --tuner_type full                   # full fine-tuning; alternatives include lora, qlora, longlora, and llamapro
    --learning_rate "${LEARNING_RATE}"
    --optim adamw_torch                 # HF optimizer; --optimizer selects plugins such as galore or muon
    --lr_scheduler_type cosine          # Any HF scheduler; pass JSON through --lr_scheduler_kwargs
    # --warmup_ratio 0.05               # Learning-rate warmup ratio (default 0)
    # LoRA (also set --tuner_type lora)
    # --lora_rank 16 --lora_alpha 32 --target_modules all-linear --lora_dropout 0.05
    # Freezing and selective unfreezing (supported by full and LoRA tuning)
    # --freeze_parameters_ratio 0.5     # Freeze the bottom 50% of parameters
    # --freeze_parameters_regex '.*layers\.0\.'   # Freeze parameters matching a regular expression
    # --trainable_parameters_regex '.*classifier.*'  # Train only matching parameters
    # LISA updates a subset of layers at each step to save memory
    # --lisa_activated_layers 2 --lisa_step_interval 20
)
# RSI_UPDATE_END

# ============================================================
# Dimension 5: steps, epochs, token budget, time, and resources
# ============================================================
# RSI_BUDGET_BEGIN
BUDGET_ARGS=(
    --num_train_epochs "${EPOCHS}"
    # --max_steps 500                   # Takes precedence over epochs when both are set
    # --max_epochs 2                    # ms-swift extension with priority over num_train_epochs
    --max_length "${MAX_LENGTH}"        # Per-sample token budget
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --gradient_checkpointing true       # Trade compute for memory
    --deepspeed zero3                   # Sixteen-GPU sharding; use FSDP instead, not together
    # --early_stop_interval 2           # Stop after two evaluation intervals without improvement
    # --resume_from_checkpoint "${RESUME_CHECKPOINT}"  # Resume a curriculum stage from its predecessor
    # --resume_only_model true          # Restore weights without optimizer state
    # --use_liger_kernel true           # Fused kernels for memory and speed
    # --use_logits_to_keep true         # Keep logits only for labeled tokens
)
# RSI_BUDGET_END

COMMON_ARGS=(
    --model "${MODEL}"
    --torch_dtype bfloat16
    --save_only_model true
    "${SWANLAB_ARGS[@]}"
)

run_sft() {
    local dataset="$1"
    local stage_output="$2"
    local resume="${3:-}"
    local stage_data=("${DATA_ARGS[@]}")
    stage_data[1]="$dataset"
    if [[ -n "$resume" ]]; then
        "${SWIFT_BIN}" sft \
            "${COMMON_ARGS[@]}" \
            --output_dir "$stage_output" \
            "${stage_data[@]}" \
            "${MIX_ARGS[@]}" \
            "${LOSS_ARGS[@]}" \
            "${OPT_ARGS[@]}" \
            "${BUDGET_ARGS[@]}" \
            --resume_from_checkpoint "$resume"
    else
        "${SWIFT_BIN}" sft \
            "${COMMON_ARGS[@]}" \
            --output_dir "$stage_output" \
            "${stage_data[@]}" \
            "${MIX_ARGS[@]}" \
            "${LOSS_ARGS[@]}" \
            "${OPT_ARGS[@]}" \
            "${BUDGET_ARGS[@]}"
    fi
}

if [[ -z "${CURRICULUM_STAGES}" ]]; then
    run_sft "${DATASET_SRC}" "${OUTPUT_DIR}"
else
    IFS=',' read -r -a stages <<< "${CURRICULUM_STAGES}"
    (( ${#stages[@]} > 0 )) || { echo "CURRICULUM_STAGES is empty" >&2; exit 1; }
    previous=""
    for index in "${!stages[@]}"; do
        dataset="${stages[$index]}"
        [[ -f "$dataset" ]] || { echo "Curriculum dataset not found: $dataset" >&2; exit 1; }
        stage_output="${OUTPUT_DIR}/curriculum-stage$((index + 1))"
        run_sft "$dataset" "$stage_output" "$previous"
        previous="$stage_output"
    done
fi
