# 自定义数据清洗（离线执行，与训练解耦）

本文介绍一种与训练完全解耦的数据清洗方式：**清洗在独立脚本中离线完成，产出干净的 jsonl；`swift sft` 只消费产物，不加载任何清洗代码**。清洗逻辑基于 `RowPreprocessor` 插件（源码位于 `swift/dataset/preprocessor/core.py`），基类负责流水线与容错，子类只写清洗规则。

三个明确的边界：

- **清洗与训练解耦**：清洗产出是一个可检查、可版本管理的 jsonl 文件，训练命令保持最简。
- **清洗不依赖 LLM 推理**：清洗是纯 Python 的行级数据变换（过滤、改写、去重、格式修复），不调用模型，不产生任何推理成本。
- **LLM 只用于打分，不用于生成**：可选地让 LLM 以"评委"方式给样本打质量分，再按阈值筛选；但**不用 LLM 生成/重写对话内容，不用它跑工具调用轨迹**。轨迹来自你的原始数据；生成类需求（采样扩充数据）走 `swift sample`，见[相关文档](#11-相关文档)。

## 1. 为什么解耦

| | 训练时插件模式 | 离线清洗模式（本文） |
| ---- | ---- | ---- |
| 清洗代码位置 | `--external_plugins` 随训练进程执行 | 独立脚本，与训练无关 |
| 训练命令 | 复杂，带插件参数 | 最简：`swift sft --dataset cleaned.jsonl` |
| 清洗成本 | 每次训练都要 map（依赖 Arrow 缓存） | 只付一次 |
| 产物 | 只存在于内存/缓存 | 干净的 jsonl，可人工抽查、可复用、可回滚 |
| 排查问题 | 混在训练日志里 | 单独运行、单独报错、单独修复 |

解耦的核心收益：清洗产出是**实体文件**而不是训练进程里的副作用——可以 diff、可以抽查、可以分享、可以多次复用给不同训练任务。

## 2. 离线清洗工作流

```
raw.jsonl
   │
   ├─(可选) LLM 打分：给每条样本打质量分，写回 score 字段
   │
   ├─ 清洗脚本（离线，纯 CPU）：RowPreprocessor 规则清洗
   │     过滤 / 改写 / 去重 / 拆条 / 按 score 阈值筛选
   │
   ▼
cleaned.jsonl ────► swift sft --dataset cleaned.jsonl（训练侧零清洗代码）
```

## 3. 清洗脚本骨架

一个可以直接运行的完整骨架（规则清洗 + 落盘）：

```python
# clean_offline.py —— 离线清洗，不启动训练、不加载模型
from typing import Any, Dict, Optional

from swift.dataset import (
    DatasetMeta, ResponsePreprocessor, register_dataset, load_dataset
)


class MyCleanPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 先走父类，把 query/response/history 拼成标准 messages
        row = super().preprocess(row)
        if row is None:
            return None

        response = row['messages'][-1].get('content')
        if not response:
            return None
        # 过滤：拒绝回答 / 过短的样本
        if '不知道' in response or len(response) < 2:
            return None
        # 改写：统一全角编号
        row['messages'][-1]['content'] = response.replace('１', '1')
        return row


register_dataset(
    DatasetMeta(
        dataset_path='/path/to/raw.jsonl',   # 或 ms_dataset_id / hf_dataset_id
        dataset_name='my_data',
        preprocess_func=MyCleanPreprocessor(),
    ))

if __name__ == '__main__':
    # strict=True：脏数据直接报错退出（体检语义），确认干净后再落盘
    train_ds, _ = load_dataset('my_data', num_proc=4, strict=True)
    train_ds.to_json('/path/to/cleaned.jsonl', lines=True, force_ascii=False)
    print(f'cleaned: {len(train_ds)} rows')
```

要点：

- **不加载模型、不占 GPU**：`preprocess_func` 是 `dataset.map` 里的纯 Python 函数，全程 CPU；
- **需要 swift 的 Python 环境**：`from swift.dataset import ...` 会间接引入 torch/transformers 等依赖，但仅 CPU 运行，不下载权重、不推理；
- `load_dataset` 返回 `(train, val)` 二元组，不切验证集时取第一个；
- `strict=True` 让清洗脚本具备"体检"语义：任何一条脏数据都当场报错，产出即干净；`strict=False` 则丢弃脏行并打印最多 10 条 traceback（见[第 8 节](#8-错误处理与容错)）。

## 4. 挂载点：清洗逻辑写在哪里

`RowPreprocessor` 提供的全部挂载点如下，离线清洗同样适用（只是在独立脚本里实例化你的预处理器时使用）：

| 挂载点 | 位置 | 粒度 | 用途 |
| ---- | ---- | ---- | ---- |
| `columns`（构造参数） | `__init__` | 列级 | 原始列名 → 标准列名映射，始终生效 |
| `preprocess(row)` | `RowPreprocessor.preprocess` | 行级 | 逐条清洗：改写 / 过滤 / 一条拆多条 |
| `prepare_dataset(dataset)` | `RowPreprocessor.prepare_dataset` | 数据集级 | 整表级操作：过滤、去重、整列变换 |
| `dataset_sample` / `random_state` / `traceback_limit`（构造参数） | `__init__` | 调试/容错 | 抽样调试、可复现随机、异常日志条数上限 |
| `standard_keys`（类属性） | `RowPreprocessor.standard_keys` | 列级 | 定义预处理后保留哪些列 |

### 4.1 行级清洗：`preprocess(row)`（最重要）

```python
def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return row
```

- 返回 `None`：丢弃该行（过滤）；
- 返回 `dict`：保留该行，可任意改写；
- 返回 `list[dict]`：一行拆成多行（数据扩增，如一条多轮对话拆成多条前缀样本）。

基类在 `preprocess` 之后才执行内建校验（`_check_messages` 等），你只需要保证输出符合标准格式，不需要自己写校验。你清洗代码里抛出的异常同样会被基类接住：`strict=True` 报错退出，`strict=False` 丢弃该行。

### 4.2 列名映射：`columns` 构造参数

```python
preprocessor = MyCleanPreprocessor(columns={'question': 'query', 'answer': 'response'})
```

`safe_rename_columns` 的特点：大小写不敏感；多个原始列映射到同一目标列时全部跳过（避免歧义覆盖）；数据集里不存在的列名静默忽略。

### 4.3 数据集级清洗：`prepare_dataset(dataset)`

整表级操作（全局去重、按统计量过滤整列）：

```python
import json

def prepare_dataset(self, dataset):
    seen = set()
    def keep(row):
        key = json.dumps(row['messages'], ensure_ascii=False)
        if key in seen:
            return False
        seen.add(key)
        return True
    return dataset.filter(keep)
```

### 4.4 构造参数：抽样与容错

| 参数 | 默认值 | 说明 |
| ---- | ---- | ---- |
| `dataset_sample` | `None` | 只抽样 N 条做预处理，调试清洗逻辑时非常有用 |
| `random_state` | `42` | 抽样/随机选择的种子，保证可复现 |
| `traceback_limit` | `10` | `strict=False` 时最多打印多少条异常 traceback，避免刷屏 |

### 4.5 列保留白名单：`standard_keys`

类属性 `standard_keys` 定义了预处理后**保留**的列（`messages`/`images`/`videos`/`audios`/`tools`/`objects` 及其 `rejected_`/`positive_`/`negative_` 前缀变体，以及 `rejected_response`、`label`、`channel`、`margin` 等）。

在离线清洗中，**不需要刻意处理这个白名单**：清洗产物落盘时保留哪些列由你的 `preprocess` 返回值决定（自定义字段如 `score` 用完即可 `pop` 掉，见[第 7 节](#7-可选步骤llm-打分筛选不生成轨迹)）。训练时才会由 `remove_unused_columns` 裁掉非标准列。

## 5. 内建预处理器与选择规则

`RowPreprocessor` 之下的核心子类（继承树见 `swift/dataset/preprocessor/`）：

| 预处理器 | 处理的数据格式 | 关键行为 |
| ---- | ---- | ---- |
| `ResponsePreprocessor` | query/response/history 格式 | 自动识别 `system`/`query`/`response` 等别名列，拼成标准 `messages` |
| `AlpacaPreprocessor` | alpaca 格式 | `instruction` + `input` 拼接成 query，`output`→response |
| `MessagesPreprocessor` | messages/sharegpt/OpenAI/Anthropic 格式 | 自动转换 `tool_calls`/`tool_use`，支持 `repair_messages` 修复字符串形式的 messages |
| `AutoPreprocessor` | 自动选择 | 有 `conversation`/`messages` 列 → `MessagesPreprocessor`；有 `instruction`+`input` → `AlpacaPreprocessor`；否则 → `ResponsePreprocessor` |

**优先继承一个与你数据格式最接近的内建预处理器**（而不是直接从 `RowPreprocessor` 写起），复用它的格式转换逻辑：

```python
from swift.dataset import AlpacaPreprocessor

class MyPreprocessor(AlpacaPreprocessor):
    def preprocess(self, row):
        # 先把 instruction/input/output 换成 query/response，
        # 此时 row 里已有标准 messages，再做自己的清洗
        row = super().preprocess(row)
        if row is None:
            return None
        ...
        return row
```

## 6. 规则清洗示例（无 LLM）

**过滤 + 改写**（直接处理 messages 格式）：

```python
class LengthFilterPreprocessor(RowPreprocessor):

    def preprocess(self, row):
        messages = row.get('messages')
        if not messages:
            return None
        last = messages[-1]
        if last.get('role') == 'assistant':
            content = last.get('content')
            if not content or len(content) < 2 or len(content) > 2000:
                return None
            last['content'] = content.strip()
        return row
```

**一条拆多条**：

```python
class SplitPreprocessor(RowPreprocessor):

    def preprocess(self, row):
        # 把一条含多轮问答的样本拆成多条"前缀 + 单轮"样本
        messages = row['messages']
        rows = []
        for i in range(2, len(messages) + 1, 2):
            rows.append({'messages': messages[:i]})
        return rows or None
```

**整表去重**：见 [4.3](#43-数据集级清洗prepare_datasetdataset) 的 `prepare_dataset` 示例。

## 7. 可选步骤：LLM 打分筛选（不生成轨迹）

### 7.1 边界

- **允许**：LLM 以"评委"方式输出一个分数或标签（如 0–5 分），用于筛选样本；
- **不允许（本文范围）**：用 LLM 生成/重写对话内容、生成工具调用轨迹、自我指导式扩充数据。清洗不做内容生产，只做内容筛选与整理。

### 7.2 模式 A（推荐）：打分是独立步骤，清洗仍无推理

把打分放在清洗**之前**的独立脚本里，打分结果写回 `score` 字段，清洗脚本只做阈值过滤。清洗本身保持无推理、可缓存、可重跑：

第一步，打分脚本（本地 vLLM 或任意 OpenAI 兼容服务，模型只输出分数）：

```python
# score_by_llm.py —— 离线打分：LLM 只输出 0-5 的整数，不生成任何内容
import json
from openai import OpenAI

client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY')  # 本地 vLLM 或兼容服务

JUDGE_PROMPT = """你是数据质量评委。请根据对话质量打分（0-5 的整数）：
- 5：完整、准确、格式规范
- 0：拒绝回答、内容空泛、格式错误
只输出一个 0-5 的整数，不要输出其他内容。

对话：
{conversation}
"""

def judge(messages):
    conversation = json.dumps(messages, ensure_ascii=False)
    resp = client.chat.completions.create(
        model='qwen3-4b',  # 换成你的模型名
        messages=[{'role': 'user', 'content': JUDGE_PROMPT.format(conversation=conversation)}],
        max_tokens=8,
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()
    try:
        score = int(''.join(ch for ch in text if ch.isdigit())[:1])
    except Exception:
        score = 0  # 解析失败按最低分处理，宁缺毋滥
    return min(max(score, 0), 5)

with open('raw.jsonl', encoding='utf-8') as fin, open('scored.jsonl', 'w', encoding='utf-8') as fout:
    for line in fin:
        row = json.loads(line)
        row['score'] = judge(row['messages'])
        fout.write(json.dumps(row, ensure_ascii=False) + '\n')
```

第二步，清洗脚本按阈值过滤，并消费掉 `score` 字段（清洗产物只留标准字段）：

```python
class ScoredCleanPreprocessor(ResponsePreprocessor):
    SCORE_THRESHOLD = 3

    def preprocess(self, row):
        row = super().preprocess(row)
        if row is None:
            return None
        score = row.pop('score', None)   # 用完即删，不进清洗产物
        if score is None or score < self.SCORE_THRESHOLD:
            return None
        return row
```

### 7.3 模式 B（不推荐）：在 `preprocess` 内直接打分

在清洗函数里调用 LLM 会破坏解耦：逐条串行推理极慢、map 重跑时重复推理、清洗逻辑与打分逻辑耦合在一起难以单独调试。如果确有此需求，请自行为推理结果做本地缓存（如按 messages 内容做 hash 落盘）。

## 8. 错误处理与容错

- **离线清洗阶段（你控制）**：
  - `load_dataset('my_data', strict=True)`：任何脏数据当场报错退出，清洗产物即干净——适合上线前体检；
  - `strict=False`：脏行被丢弃，最多打印 `traceback_limit`（默认 10）条 traceback；`MaxLengthError`（超长样本）永远静默丢弃；
  - 丢弃发生时日志输出 `Dataset filtered, origin length: X, filtered dataset length: Y`，用这个数字核对丢了多少条。
- **训练阶段（官方管线兜底，与你无关）**：即使喂的是清洗产物，`swift sft` 仍会再过一遍官方管线——`--strict false`（默认）在预处理阶段丢弃坏样本，编码阶段由 `LazyLLMDataset` 自动换一条（最多重试 10 次）。两道防线不冲突，也不会因为"已经洗过"而失效。

## 9. 训练侧用法（极简）

```bash
swift sft --dataset /path/to/cleaned.jsonl
```

- 无需 `--external_plugins`、无需 `--custom_dataset_info`——训练侧零清洗代码；
- 清洗产物本身就是标准格式（messages 等），官方管线直接消费；
- 若某次清洗配置想固化下来（复跑同一种清洗），保留清洗脚本并随产物一起归档即可。

## 10. 调试建议

1. **独立脚本单测**：清洗脚本的 `__main__` 就是天然的单测入口，`load_dataset` 直接看输出，不依赖训练链路；
2. **抽样调试**：实例化预处理器时传 `dataset_sample=100`，先抽样验证清洗逻辑；
3. **抽查产物**：`cleaned.jsonl` 落盘后直接 `head`/diff 检查，比在训练日志里翻找直观得多；
4. **核对丢弃比例**：观察 `Dataset filtered` 日志，判断过滤阈值是否合理；
5. **缓存陷阱**：预处理结果有 Arrow 缓存，改完清洗逻辑后若怀疑命中旧缓存，`load_dataset(..., load_from_cache_file=False)` 重跑或删除缓存目录。

## 11. 相关文档

- [自定义数据集](../Customization/Custom-dataset.md)：数据集格式与接入方式总览；
- [训练过程控制维度指南](Training-controls.md)：数据清洗在整个训练控制体系中的位置；
- [sample 数据生成](Sample.md)：需要生成/扩充数据（而非筛选）时使用；
- 源码参考：`swift/dataset/preprocessor/core.py`（基类与内建预处理器）、`swift/dataset/preprocessor/extra.py`（文本生成/分类示例）、`swift/dataset/register.py`（注册机制）。
