# Custom Data Cleaning

Data cleaning is an offline, deterministic preprocessing step. It produces a reviewable JSONL artifact before `swift sft` starts; training must never import or execute cleaning code.

## Contract

- Read one JSON object per input line and preserve valid `messages`, `tools`, and task metadata unless a documented repair is required.
- Validate the conversational schema and role ordering.
- Normalize only deterministic representation defects such as malformed whitespace or structurally equivalent empty fields.
- Remove invalid rows and exact duplicates with explicit counters and stable rules.
- An LLM judge may only assign a score or label when the Task Definition explicitly authorizes its cost. It must never generate or rewrite semantic content or tool trajectories.
- Preserve input order unless the task explicitly authorizes ordering changes; ordering and curriculum belong to RecipeAgent.
- A row may be filtered, repaired, deduplicated, or split according to a fixed rule. Preserve every existing per-message `loss` field; supervision belongs to ObjectiveAgent.
- Write UTF-8 JSONL deterministically and report input rows, output rows, removal counts, repair counts, duplicate counts, and split counts.

## RowPreprocessor model

ms-swift's `RowPreprocessor` pattern separates row transformation from pipeline error handling. A cleaner should implement a pure row mapping/filtering function, return a valid normalized row or reject it, and collect bounded diagnostics. The standalone candidate build may use Python's standard library to apply the same design without importing the full ms-swift runtime.

## Safety and reproducibility

The cleaner runs without network access, does not mutate `/base`, and writes only declared outputs under `/candidate`. The controller-owned `/candidate/build.sh` and `/candidate/candidate.json` are already present and are not candidate outputs. Use fixed constants for every threshold. Fail closed on malformed JSON and make every removal reason auditable in candidate statistics. Infer valid roles and fields from the actual ms-swift corpus; roles such as `tool_call` are valid and must not be rejected by a hard-coded OpenAI-only allowlist.
