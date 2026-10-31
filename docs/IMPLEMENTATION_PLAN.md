# Implementation plan

## Objective

Answer whether fine-tuning Whisper tiny English on short medical-symptom speech improves transcription of unseen phrases. Report both overall ASR quality and exact symptom-term recall, with honest limitations. The current code is a starting point, not a completed study.

## Milestones and acceptance checks

| Stage | Work | Acceptance check | Status |
| --- | --- | --- | --- |
| 1. Data audit | Download pinned dataset revision; inspect columns, durations, clipping, empty audio, transcript labels, and original source terms. | Save counts and a review log; confirm redistribution terms before releasing any derived data. | Pending external data access |
| 2. Split protocol | Make deterministic transcript-disjoint train/validation/test manifest. | No normalized phrase crosses partitions; record clip and unique-phrase counts; freeze manifest. | Implemented, untested on full dataset |
| 3. Baseline | Run pretrained Whisper tiny English on validation. | Save full predictions, normalized WER/CER, term support/recall, hardware, model SHA, dataset SHA, code commit. | Implemented, not executed |
| 4. Fine-tuning | Run a short smoke train, then full training on train only; select by validation WER. | No test data is read; checkpoint loads for inference; memory fits 6 GB or batch settings are adjusted and logged. | Implemented, not executed |
| 5. Final comparison | Freeze model, term list, normalization, and exclusions; score both checkpoints on full test once. | Same test rows, decoding settings, and metric code; report absolute and relative changes with sample counts. | Pending |
| 6. Robustness | Add reproducible noise at fixed SNRs to test audio without changing text; compare paired clean/noisy results. | Seeded transformations, documented noise source, audio peak checks, and no noise applied to validation while tuning. | Planned |
| 7. Uncertainty and errors | Add transcript-group bootstrap confidence intervals and a paired model comparison; audit errors by term and audio quality. | Report intervals, number of unique phrases, examples, and failures; distinguish exploratory from confirmatory analyses. | Planned |
| 8. Public release | Review README, RESULTS, attribution, dependency versions, CI, and ignored files. | `git status` contains no audio, checkpoints, tokens, or private data; a new machine can follow setup. | Scaffold only |

## Recommended execution order

1. Finish data provenance review and dataset audit before training.
2. Run all commands in the README, starting with limited smoke checks.
3. Inspect validation errors; make only documented protocol changes.
4. Freeze settings, run the full test, fill in `RESULTS.md`, and review sample predictions.
5. Add the robustness and bootstrap analyses as a second documented experiment. Do not silently mix their results into the original test claim.

## Resume gate

Only after completed runs: describe the dataset, model, exact change in WER or term recall, and reproducibility work. If fine-tuning does not improve the baseline, report that honestly and explain the error analysis. Do not claim clinical deployment or patient safety.
