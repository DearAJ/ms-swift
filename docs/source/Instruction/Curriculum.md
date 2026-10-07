# 课程学习（Curriculum）实战：难度分桶 + 分阶段续训

本指南回答一个问题：**怎么让模型"先学简单、再学难"？** 落地方案采用 [训练过程控制维度指南](Training-controls.md) 里的方案③（分阶段多次训练、checkpoint 续训），难度划分采用**规则打分**（不调 LLM、纯 CPU、秒级完成）。

三条替代方案的完整对比见 [Training-controls.md](Training-controls.md) 维度 2；如果想用 LLM 评委打难度分替代规则打分，见 [自定义数据清洗](Custom-data-cleaning.md) 第 7 节。

## 1. 什么是课程学习

课程学习就是模仿人类上课：**先给模型简单的任务，再逐步上难度**，而不是把所有难度的数据混在一起随便喂。对 Agent SFT 而言，"难度"通常体现在：对话轮数、工具调用次数、回复长度。

## 2. 为什么选"规则打分 + 分阶段续训"

| 候选做法 | 问题 |
| ---- | ---- |
| 预排序 + `--dataset_shuffle false` | 每个 epoch 顺序都一样（从易到难重复），且难度分档无法中途检查 |
| 自定义 `--callbacks` 回调换数据 | 要写 plugin 代码，训练进程内调试麻烦 |
| **分阶段多次训练 + checkpoint 续训** | **每阶段之间可以停下来评估模型、调整下一阶段数据，最可控** |

难度打分选规则而非 LLM：Agent 数据的难度与轮数 / 工具调用次数强相关，规则可解释、零成本、可复现；LLM 打分（见 [Custom-data-cleaning.md](Custom-data-cleaning.md) 第 7 节）只在规则分不出的场景才需要。

## 3. 工作流总览

```
tb2_sft.jsonl
   │
   ├─ ① 难度打分分桶（split_difficulty.py，纯 CPU）
   │     按 对话轮数 / 工具调用次数 / 长度 加权打分，
   │     排序后切三份，产出 easy.jsonl / mid.jsonl / hard.jsonl
   │
   ├─ ② 三阶段训练（train_qwen35_agent_sft_8gpu.sh）
   │     stage1: easy.jsonl   →  output/curriculum-stage1
   │     stage2: mid.jsonl    →  output/curriculum-stage2（续训自 stage1）
   │     stage3: hard.jsonl   →  output/curriculum-stage3（续训自 stage2）
   ▼
模型按"易 → 中 → 难"完成 Agent SFT
```

## 4. 第一步：难度打分分桶

### 4.1 打分规则

对每条对话按下式计算难度分（权重可按数据特点调整）：

```
难度分 = 对话轮数 × 2 + 工具调用次数 × 3 + （总长度 > 2000 字符 ？ 2 : 0）
```

- 对话轮数 = `messages` 中 `user` 角色的数量；
- 工具调用次数 = `messages` 中 `tool_call` 角色的数量；
- 总长度 = 所有消息 `content` 的字符数之和。

### 4.2 分桶脚本

```python
# split_difficulty.py —— 规则打分，产出 easy/mid/hard 三个 jsonl（纯 CPU，秒级）
import json
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else 'input/tb2_sft.jsonl'
OUT_EASY, OUT_MID, OUT_HARD = 'input/easy.jsonl', 'input/mid.jsonl', 'input/hard.jsonl'

def score(row):
    msgs = row['messages']
    turns = sum(1 for m in msgs if m['role'] == 'user')                    # 对话轮数
    tool_calls = sum(1 for m in msgs if m['role'] == 'tool_call')          # 工具调用次数
    length = sum(len(str(m.get('content', ''))) for m in msgs)             # 总长度
    return turns * 2 + tool_calls * 3 + (2 if length > 2000 else 0)

rows = [json.loads(line) for line in open(SRC, encoding='utf-8')]
for row in rows:
    row['_score'] = score(row)
rows.sort(key=lambda r: r['_score'])        # 从易到难排好

n = len(rows)
buckets = {
    OUT_EASY: rows[:n // 3],
    OUT_MID:  rows[n // 3:2 * n // 3],
    OUT_HARD: rows[2 * n // 3:],
}

for path, data in buckets.items():
    with open(path, 'w', encoding='utf-8') as f:
        for row in data:
            row.pop('_score', None)         # 打分用完即删，产物只留标准字段
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'{path}: {len(data)} 条，难度分范围 '
          f'{score(data[0])} ~ {score(data[-1])}')
```

在 RSI 多 Agent 流程中，RecipeAgent 的分桶产物必须与输入数据逐行守恒：每条输入恰好进入一个桶、桶内保持原顺序、不得清洗或改写样本。若需要过滤坏数据，应由 DataAgent 生成独立候选，再由主 Agent 决定是否组合。

运行后**先检查每桶的分数范围**是否合理。如果希望按更直观的规则分档（比如"没有工具调用算 easy、1~2 次算 mid、3 次以上算 hard"），把排序三等分改成固定阈值过滤即可。

## 5. 第二步：三阶段训练

使用现有训练脚本 [train_qwen35_agent_sft_8gpu.sh](../../scripts/train_qwen35_agent_sft_8gpu.sh)（训练机上为 `scripts/train_qwen35_agent_sft_8gpu.sh`）。脚本已预留续训接口 `RESUME_CHECKPOINT`（当前为注释状态，启用方法见 5.2）。

### 5.1 阶段命令

```bash
# 阶段 1：只喂简单数据（与普通训练完全一样）
OUTPUT_DIR=output/curriculum-stage1 \
  ./scripts/train_qwen35_agent_sft_8gpu.sh input/easy.jsonl

# 阶段 2：喂中等数据，接着 stage1 的成果继续练
RESUME_CHECKPOINT=output/curriculum-stage1 \
  OUTPUT_DIR=output/curriculum-stage2 \
  ./scripts/train_qwen35_agent_sft_8gpu.sh input/mid.jsonl

# 阶段 3：喂难数据，接着 stage2 继续练
RESUME_CHECKPOINT=output/curriculum-stage2 \
  OUTPUT_DIR=output/curriculum-stage3 \
  ./scripts/train_qwen35_agent_sft_8gpu.sh input/hard.jsonl
```

`RESUME_CHECKPOINT` 指向**上一阶段的输出目录**（无需指定具体的 `checkpoint-xxx`，训练器会自动挑其中最新的一个）。

### 5.2 脚本接口（预留，注释状态）

训练脚本中已按注释预留接口，启用只需两步：

1. 取消变量定义注释（头部"Curriculum 接口"区）：
   ```bash
   RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"   # 上一阶段 OUTPUT_DIR
   ```
2. 取消参数注释（维度 5 `BUDGET_ARGS` 内）：
   ```bash
   --resume_from_checkpoint "${RESUME_CHECKPOINT}"
   ```

接口留空时脚本行为与现在完全一致（不续训）；传入路径时才追加续训参数。

### 5.3 为什么换数据集不会冲突

续训时训练默认 `load_data_args=False`：不会把上一阶段 checkpoint 里记录的数据集参数加载回来，本次新传的 `--dataset` 正常生效。想恢复优化器状态时不要加 `--resume_only_model`（它只加载权重、丢弃优化器状态）。

## 6. 注意事项

- **验证集**：每个阶段都会从本阶段数据里随机切 1% 做验证（`--split_dataset_ratio 0.01`），不同阶段的验证集不同，指标不可直接横向对比；
- **优化器状态**：当前脚本 `--save_only_model true` 保存的 checkpoint 以权重为主，跨阶段续训主要继承模型参数；若希望连优化器状态一起继承，需去掉该参数（磁盘占用会变大）；
- **实验命名**：三个阶段共用同一个 `SWANLAB_PROJECT` 会混在一起，建议每阶段用不同 `SWANLAB_EXP_NAME`（如 `curriculum-stage1/2/3`）；
- **每阶段预算可不同**：`EPOCHS`、`LEARNING_RATE` 等都是环境变量，可每阶段单独覆盖，例如难数据阶段给更小学习率；
- **分桶比例可调**：若 easy 桶占比太大（数据整体偏简单），可按难度分阈值切（如按 tool_call 次数 0 / 1-2 / 3+ 分档），不必三等分。

## 7. 常见问题

**Q：难度规则怎么定才合理？**
先用本节默认规则跑一版，人工抽查三桶各 20 条，看"简单/难"是否符合直觉；不符合就调权重或换特征（如换成工具调用失败次数、回复 token 数）。

**Q：规则分不出来，想用 LLM 打难度分？**
把 [Custom-data-cleaning.md](Custom-data-cleaning.md) 第 7 节的 LLM 评委脚本的 prompt 从"质量分"改成"难度分"（0-5），打分结果分桶即可。注意 LLM 打分需要推理服务、按条数计耗时，建议先小批试点。

**Q：训练中途中断了怎么办？**
中断阶段重新执行同一条命令即可：`RESUME_CHECKPOINT` 指向本阶段自己的输出目录，会从该阶段最新的 checkpoint 继续。
