# 训练过程控制维度指南

本指南按五个维度介绍 ms-swift 对训练过程的控制能力：训练数据、数据组成与采样、损失与监督、优化器与参数更新、训练预算与资源。每一维度的控制项要么是现成的命令行参数，要么通过插件机制（`swift/loss/mapping.py`、`swift/optimizers/mapping.py`、`swift/loss_scale/mapping.py`、`swift/callbacks/mapping.py` 中的注册表，均提供 "Add your own" 扩展点）接入，无需修改框架核心代码。

## 1. 数据内容、质量、筛选、清洗

| 能力 | 参数 / 机制 | 说明 |
| ---- | ----------- | ---- |
| 单条数据内容与格式 | `--dataset`、`--columns`、`--custom_dataset_info` | 支持 jsonl/csv/json/文件夹/HF 数据集；`--columns` 做列映射（如 `{"text1": "query", "text2": "response"}`）；`--custom_dataset_info` 注册自定义数据集格式 |
| 质量筛选 | `--strict`、`--truncation_strategy` | `--strict false` 自动丢弃预处理失败的样本；超长样本按 `delete/left/right/split` 策略处理 |
| 训练时过滤 | `LazyLLMDataset`、`--cached_dataset` | 训练时惰性 tokenize，自动跳过失败样本；`swift export --to_cached_dataset` 生成缓存时过滤错误样本 |
| 数据清洗 | 自定义 `RowPreprocessor` 插件 | 注册自定义预处理器做离线清洗（与训练解耦，挂载点见[自定义数据清洗](Custom-data-cleaning.md)）；无自研预清洗脚本，坏数据统一由 `--strict` 兜底（false=丢弃并记日志，true=报错退出） |

局限：无内建的自动质量打分 / 去重工具，需要自定义 preprocessor 或外部脚本完成，但插件接口是现成的。

## 2. 数据比例、采样、顺序、curriculum、数据组成

| 能力 | 参数 / 机制 | 说明 |
| ---- | ----------- | ---- |
| 数据比例与组成 | 数据集语法 `dataset_id:subset#count`、`--interleave_prob` | `--dataset d1#5000 d2#3000` 精确控制各数据集条数；`--interleave_prob 0.3 0.7` 按概率混合；`--stopping_strategy` 控制混合耗尽策略 |
| 采样 | `--data_seed`、`--dataset_shuffle`、`--shuffle_buffer_size` | `#count` 支持无放回随机采样与重复采样；streaming 模式下由 `--shuffle_buffer_size` 控制混洗窗口 |
| 顺序 | `--dataset_shuffle false`、`--group_by_length` | 关闭混洗可保持原始顺序；按长度分组减少 padding 浪费 |
| 数据组成 | `--split_dataset_ratio`、`--packing`、`--padding_free`、`--streaming`、`--cached_dataset` | 自动划分验证集；packing 打包提升 token 利用率；streaming 边读边训（必须配合 `--max_steps`） |
| Curriculum | 无内建参数 | 三种替代方案：① 外部按难度预排序 + `--dataset_shuffle false` 保序；② 自定义 `--callbacks` 回调分阶段换数据；③ 分阶段多次训练（checkpoint 续训）；落地路线见[课程学习实战](Curriculum.md)（规则难度分桶 + 分阶段续训） |

## 3. Loss、Mask、监督范围、训练信号、Loss weighting

| 能力 | 参数 / 机制 | 说明 |
| ---- | ----------- | ---- |
| Loss 类型 | `--loss_type` | 内建 `cross_entropy`、`cosine_similarity`、`contrastive`、`online_contrastive`、`infonce`、`pointwise_reranker`、`listwise_reranker`，可在 `swift/loss/mapping.py` 注册自定义损失 |
| Mask 与监督范围 | 模板机制、`--loss_scale` | 模板自动只对 response 部分计算损失；`--loss_scale` 提供基础策略 `default / last_round / all` 及细粒度策略 `react`、`hermes`、`qwen`、`agentflan`、`alpha_umi`、`ignore_empty_think`、`ignore_think_prefix` |
| 逐条消息监督 | 数据中逐消息 `loss` 字段 | 每条 message 可显式声明是否参与损失计算；不写时默认仅 assistant（含折叠后的 tool_call）参与 |
| Loss weighting | `--loss_scale` 链式组合、`--mrl_dims` | 多个 loss_scale 以 `+` 连接，权重相乘，如 `last_round+hermes+ignore_empty_think`；`--mrl_dims '{"32": 1.0, "64": 1.0}'` 做多维度 embedding 加权损失 |
| 训练信号 | GRPO/RLHF 奖励、`--router_aux_loss_coef`、`--enable_dft_loss`、`--enable_channel_loss` | 强化学习任务的 reward 系统；MoE 辅助损失系数；DFD / channel 附加损失 |

## 4. Optimizer、LR、Scheduler、PEFT、Layer/Module 更新范围

| 能力 | 参数 / 机制 | 说明 |
| ---- | ----------- | ---- |
| Optimizer | `--optim`、`--optimizer` | 继承 HF 全部优化器（`adamw_torch`、`adamw_torch_fused`、`adafactor` 等）；`--optimizer` 选择插件：`galore`、`lorap`、`muon`、`muonclip`、`multimodal`，各自带完整参数组（如 `--galore_rank 128 --galore_update_proj_gap 50`） |
| 学习率 | `--learning_rate`、`--aligner_lr`、`--vit_lr`、`--warmup_ratio` / `--warmup_steps` | 多模态模型可给 aligner / ViT 设置独立学习率（自动切换到 multimodal optimizer） |
| Scheduler | `--lr_scheduler_type`、`--lr_scheduler_kwargs` | 默认 `cosine`，支持 HF 全部调度器；`--lr_scheduler_kwargs` 以 JSON 传参 |
| PEFT | `--tuner_type`、`--target_modules`、`--target_regex`、`--tuner_backend` | `lora / qlora / longlora / llamapro / neftune / part / scetuning / reft / adapter / prompt / side / restuning`，`full` 表示全参微调；backend 可选 swift/peft |
| Layer/Module 更新范围 | `--freeze_parameters`、`--freeze_parameters_regex`、`--freeze_parameters_ratio`、`--freeze_llm / freeze_vit / freeze_aligner`、`--trainable_parameters(_regex)`、LISA | 按前缀 / 正则 / 比例（自底层向上冻结 0~1）冻结参数，或用 `trainable_*` 反向解冻；`--lisa_activated_layers` + `--lisa_step_interval` 实现每步只更新部分层 |

## 5. Steps、Epochs、Token budget、训练时间、资源预算

| 能力 | 参数 / 机制 | 说明 |
| ---- | ----------- | ---- |
| Steps | `--max_steps`、`--gradient_accumulation_steps` | GA 留空时自动计算 `ceil(16 / batch_size / world_size)`，保证有效全局 batch |
| Epochs | `--num_train_epochs`、`--max_epochs` | `--max_epochs` 为 ms-swift 扩展，优先级高于 `num_train_epochs` |
| Token budget | `--max_length`、`--packing`、`--packing_length` | 单样本 token 上限；packing 消除 padding 浪费；全局 token 预算可由 `steps × batch_size × max_length` 精确换算，streaming 模式必须配合 `--max_steps` |
| 训练时间 | `--early_stop_interval`、`--resume_only_model` | 指标不提升自动早停；断点恢复时选择是否加载优化器状态 |
| 资源预算 | `--per_device_train_batch_size`、`--deepspeed`、`--fsdp`、`--gradient_checkpointing`、`--use_liger_kernel`、`--use_logits_to_keep`、Megatron | 显存 / 速度 / 并行度控制，支持 DeepSpeed（含 zero3）、FSDP 与 Megatron（TP/PP/CP/EP） |

## 6. 实战：train_qwen35_agent_sft_8gpu.sh 使用了哪些控制

`scripts/train_qwen35_agent_sft_8gpu.sh` 是一个全参 Agent SFT 训练脚本，它只用到上述能力的一个子集：

**训练前预处理：**

> 无自研脚本：ms-swift 官方管线自动折叠 tool_call、合并同角色消息，默认仅对 response 部分计算损失，无需逐条写 `loss` 标记或预清洗；坏数据由 `--strict false` 在预处理阶段丢弃并记日志。

**swift sft 实际使用的参数：**

| 维度 | 用到的控制 | 未用到（走默认值） |
| ---- | ---------- | ------------------ |
| 数据 | `--split_dataset_ratio 0.01`（划分验证集） | 多数据集比例、`#count` 采样、`--interleave_prob`、curriculum、packing |
| 损失与监督 | 无显式控制，走默认策略：仅 response 部分计算损失 | `--loss_type`（默认 cross_entropy）、`--loss_scale`（默认 default） |
| 优化与更新 | `--tuner_type full`（全参更新全部参数）、`--learning_rate 5e-6` | PEFT、`--freeze_*`、LISA、optimizer 插件、scheduler 显式设置（默认 cosine + AdamW） |
| 预算与资源 | `--num_train_epochs`、`--max_length 16384`（token 预算）、`--gradient_accumulation_steps 2`、`--per_device_train_batch_size 1`、`--gradient_checkpointing true`、`--deepspeed zero3`（8 GPU 资源控制） | `--max_steps`、`--max_epochs`、`--early_stop_interval` |

即：该脚本覆盖了坏数据兜底（`--strict false`）、更新范围（全参）、学习率、Epochs / Token 预算 / 资源预算；监督范围走默认策略（仅 response 部分计算损失），但**没有使用**多数据集混合、curriculum、loss_type/loss_scale 定制、逐消息 `loss` 标记、PEFT、冻结/LISA 等控制能力——这些控制项留给其他训练任务按需开启。
