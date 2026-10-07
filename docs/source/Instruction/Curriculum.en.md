# Curriculum Learning

This guide implements curriculum learning with deterministic difficulty buckets and staged continuation.

## Difficulty scoring

Use transparent, stable features available in each record, such as message count, tool-call count, assistant response length, or total character/token proxy length. Do not use an LLM judge. Define fixed thresholds before processing and report the bucket counts.

## Bucket generation

Read the baseline JSONL once and assign every valid row to exactly one of `easy`, `medium`, or `hard`. Preserve every field and value in each row and preserve original order within each bucket; JSON whitespace and key order may change. The three outputs together must contain exactly the same row multiset as the input. Filtering belongs to DataAgent and must be a separate candidate; RecipeAgent does not clean or rewrite rows.

## Staged training

Train easy, then medium, then hard. Each stage uses the previous stage checkpoint as its resume source. Set `--dataset_shuffle false` when strict within-bucket order matters. Keep objective, optimizer, and budget controls unchanged; those belong to other specialists.

## Candidate requirements

RecipeAgent may output curriculum datasets under `input/curriculum/` and may modify only the `RSI_RECIPE` block of the training script. It must report bucket thresholds, counts, row conservation, ordering behavior, and the stage sequence. The build must be deterministic and offline.
