#!/usr/bin/env bash
# Full-parameter Agent SFT with an explicit five-dimension control panel.
# Every knob below maps to a section of docs/source/Instruction/Training-controls.md.
# Lines starting with `# --xxx` are optional knobs: un-comment them to enable.
#
# Usage:
#   MODEL=/path/to/Qwen3.5-9B ./scripts/train_qwen35_agent_sft_8gpu.sh
#   # 默认数据: ms-swift/input/tb2_sft.jsonl，默认输出: ms-swift/output/agent-sft
#   # 覆盖示例: OUTPUT_DIR=output/exp2 ./scripts/train_qwen35_agent_sft_8gpu.sh other.jsonl
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"   # ms-swift 根目录

DATASET_SRC="${1:-${PROJECT_DIR}/input/tb2_sft.jsonl}"
MODEL="${MODEL:?Set MODEL to the local model directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/agent-sft}"

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

SWIFT_BIN="${SWIFT_BIN:-swift}"

# ── Curriculum 接口（暂未启用，详见 docs/source/Instruction/Curriculum.md）──
# 启用步骤：
# 按"易 → 中 → 难"分三次运行本脚本（每次换 DATASET_SRC 与 OUTPUT_DIR）。
# RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"   # 上一阶段 OUTPUT_DIR，如 output/curriculum-stage1

# ── 数据源列表：多数据集混合就追加条目（维度 2 的 --interleave_prob 配合使用）──
DATASETS=("${DATASET_SRC}")
# DATASETS+=("data/tb2_math.jsonl#2000")                 # 本地文件，只取 2000 条
# DATASETS+=("hf::meta-llama/Llama-3.2-1B-Instruct#1000")  # HuggingFace 仓库数据集

for ds in "${DATASETS[@]}"; do
    [[ "${ds}" == *"::"* ]] && continue                 # 仓库数据集跳过本地检查
    ds_path="${ds%%#*}"                                  # 去掉 #count 后缀
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
# 维度 1：数据内容 / 质量 / 筛选 / 清洗
# ============================================================
DATA_ARGS=(
    --dataset "${DATASETS[@]}"          # 数据源（jsonl/csv/文件夹/HF 数据集）
    --strict false                      # false=坏数据丢弃+有限日志（默认）；true=报错退出
    --truncation_strategy delete        # 超长样本处理：delete(删)/left(截头)/right(截尾)/split(拆条)
    # --columns '{"text1": "query", "text2": "response"}'  # 列名映射到标准格式
    # --custom_dataset_info data/dataset_info.json          # 注册自定义数据集格式
    # --cached_dataset true              # 用 `swift export --to_cached_dataset` 预生成的缓存
)
# 自定义逐条清洗（离线、与训练解耦）：写 Python 插件继承 RowPreprocessor，
# 用独立脚本清洗产出干净 jsonl 后再喂训练（不随 swift sft 进程执行），
# 挂载点与示例见 docs/source/Instruction/Custom-data-cleaning.md。

# ============================================================
# 维度 2：数据比例 / 采样 / 顺序 / curriculum / packing
# ============================================================
MIX_ARGS=(
    --split_dataset_ratio 0.01          # 随机切 1% 做验证集
    --data_seed 42                      # 混洗种子（固定后可复现）
    --dataset_shuffle true              # false=保持文件原始顺序（curriculum 前提）
    # --interleave_prob 0.3 0.7         # 多数据集按概率混合（长度需等于数据集个数）
    # --stopping_strategy first_exhausted  # 混合耗尽策略：first_exhausted/all_exhausted
    # --shuffle_buffer_size 1000        # streaming 模式的混洗窗口大小
    # --group_by_length true            # 同批次按长度分组，减少 padding 浪费
    # --packing true --packing_length 16384  # 多条样本塞进一条序列，提升 token 利用率
    # --padding_free true               # 免 padding，显存/速度优化（需模板支持）
    # --streaming true                  # 边读边训（必须配合 --max_steps）
)
# curriculum（无内建参数，三种做法）：
#   ① 数据先按难度排序，再 --dataset_shuffle false 保持顺序（最简单）；
#   ② 自定义 callbacks 分阶段换数据；
#   ③ 分阶段多次训练，用上一阶段 checkpoint 续训。
# 落地路线（③ + 规则难度分桶）见 docs/source/Instruction/Curriculum.md，
# 脚本对应接口：头部 RESUME_CHECKPOINT 变量 + 维度 5 的 --resume_from_checkpoint。

# ============================================================
# 维度 3：Loss / Mask / 监督范围 / Loss weighting
# ============================================================
LOSS_ARGS=(
    --loss_type cross_entropy           # 内建：cosine_similarity/contrastive/infonce/pointwise_reranker...
    --loss_scale default                # default=只对 assistant 回答计损；last_round=只最后一轮；all=全部
    # --loss_scale last_round+hermes    # 链式组合（权重相乘）；细粒度：react/qwen/agentflan/ignore_empty_think
    # --mrl_dims '{"32": 1.0, "64": 1.0}'  # 多维度 embedding 加权损失
)
# 逐条消息计损：直接在数据里给某条 message 加 "loss": 0/1 字段（无需参数）。

# ============================================================
# 维度 4：Optimizer / LR / Scheduler / PEFT / 更新范围
# ============================================================
OPT_ARGS=(
    --tuner_type full                   # full=全参微调；lora/qlora/longlora/llamapro/... 见下
    --learning_rate "${LEARNING_RATE}"
    --optim adamw_torch                 # HF 内置优化器；--optimizer 选插件：galore/lorap/muon/muonclip
    --lr_scheduler_type cosine          # HF 全部调度器可用；--lr_scheduler_kwargs 传 JSON
    # --warmup_ratio 0.05               # 学习率预热比例（默认 0）
    # ── LoRA（需同时改 --tuner_type lora）──
    # --lora_rank 16 --lora_alpha 32 --target_modules all-linear --lora_dropout 0.05
    # ── 冻结 / 反向解冻（full 和 lora 下均可用）──
    # --freeze_parameters_ratio 0.5     # 自底层向上冻结 50% 的参数
    # --freeze_parameters_regex '.*layers\.0\.'   # 按正则冻结（如只冻结第 0 层）
    # --trainable_parameters_regex '.*classifier.*'  # 反向：只解冻匹配的部分
    # ── LISA（每步只更新部分层，省显存）──
    # --lisa_activated_layers 2 --lisa_step_interval 20
)

# ============================================================
# 维度 5：Steps / Epochs / Token 预算 / 训练时间 / 资源
# ============================================================
BUDGET_ARGS=(
    --num_train_epochs "${EPOCHS}"
    # --max_steps 500                   # 与 epochs 二选一（两者都传时 max_steps 优先）
    # --max_epochs 2                    # ms-swift 扩展，优先级高于 num_train_epochs
    --max_length "${MAX_LENGTH}"        # 单样本 token 上限（Token 预算）
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --gradient_checkpointing true       # 用计算换显存
    --deepspeed zero3                   # 8 卡资源控制；或换 --fsdp full_shard auto_wrap（二选一）
    # --early_stop_interval 2           # 指标连续 2 个评估周期不提升则早停
    # --resume_from_checkpoint "${RESUME_CHECKPOINT}"  # Curriculum 方案③：从上一阶段 OUTPUT_DIR 续训（配合头部变量启用）
    # --resume_only_model true          # 断点恢复时只加载权重，不加载优化器状态
    # --use_liger_kernel true           # Liger 融合算子，显存/速度优化
    # --use_logits_to_keep true         # 只保留标签部分的 logits，省显存（训练时默认自动开启）
)

COMMON_ARGS=(
    --model "${MODEL}"
    --output_dir "${OUTPUT_DIR}"
    --torch_dtype bfloat16
    --save_only_model true
    "${SWANLAB_ARGS[@]}"
)

"${SWIFT_BIN}" sft \
    "${COMMON_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${MIX_ARGS[@]}" \
    "${LOSS_ARGS[@]}" \
    "${OPT_ARGS[@]}" \
    "${BUDGET_ARGS[@]}"
