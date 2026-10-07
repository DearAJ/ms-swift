# Training Control Dimensions

This guide defines the five independent control dimensions used by the RSI specialists. An experiment should change one dimension at a time unless the Main Agent explicitly selects several validated candidates.

## 1. Data content and quality

`DataAgent` owns data content and quality. It may validate schemas, repair deterministic formatting defects, remove invalid or duplicate rows, apply reproducible row-level filters, or split rows when explicitly justified. It also owns data preprocessing controls inside `RSI_DATA`, including strictness, truncation strategy, and column mapping. It must preserve every per-message `loss` field and must not change mixture, order, objective, optimizer, or compute budget.

Relevant ms-swift controls include `--dataset`, `--columns`, `--custom_dataset_info`, `--strict`, `--truncation_strategy`, and cached datasets. For custom offline cleaning, follow `Custom-data-cleaning.en.md`.

## 2. Mixture, sampling, order, and curriculum

`RecipeAgent` owns dataset sampling and curriculum. It may change `--interleave_prob`, `--stopping_strategy`, `--data_seed`, `--dataset_shuffle`, `--shuffle_buffer_size`, `--group_by_length`, `--packing`, and `--padding_free`. It may create deterministic curriculum bucket datasets and arrange staged checkpoint continuation. It must not rewrite sample content or change objective, optimizer, or budget controls.

For a staged curriculum, follow `Curriculum.en.md`.

## 3. Objective and supervision

`ObjectiveAgent` owns loss and supervision controls. Relevant controls include `--loss_type`, `--loss_scale`, per-message `loss` fields, and supported objective-specific weighting. It may modify `RSI_OBJECTIVE` and/or emit the configured dataset with only per-message `loss` changes. It must preserve row order and all non-loss content and must not change data composition, optimizer settings, or compute budget.

## 4. Parameter updates

`UpdateAgent` owns optimizer and update-policy controls inside `RSI_UPDATE`. Relevant controls include `--tuner_type`, `--learning_rate`, `--optim`, `--lr_scheduler_type`, warmup, LoRA settings, freezing expressions, and LISA. Batch size and gradient accumulation belong to BudgetAgent. UpdateAgent must not change data, objective semantics, or total compute budget.

## 5. Compute budget and resources

`BudgetAgent` owns steps, epochs, sequence length, batch/accumulation budget, checkpoint cadence, memory controls, and resource limits. Relevant controls include `--num_train_epochs`, `--max_steps`, `--max_length`, batch sizes, `--gradient_accumulation_steps`, `--gradient_checkpointing`, DeepSpeed/FSDP, early stopping, and checkpoint resume controls.

## Ownership rule

The training script uses `RSI_DATA`, `RSI_RECIPE`, `RSI_OBJECTIVE`, `RSI_UPDATE`, and `RSI_BUDGET` marker pairs. A specialist may modify only its matching block. DataAgent may also emit the configured dataset; RecipeAgent may emit byte-preserving curriculum buckets under `input/curriculum/`; ObjectiveAgent may emit the configured dataset only when all non-loss content is unchanged. Unchanged files must not be copied or listed as outputs. Docker validation enforces these semantic boundaries.
