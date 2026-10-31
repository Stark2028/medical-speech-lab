# Implementation plan

## Objective

Answer whether fine-tuning Whisper tiny English on short medical-symptom speech improves transcription of unseen phrases. Report both overall ASR quality and exact symptom-term recall, with honest limitations. The validation baseline is measured; the fine-tuned and final test comparison is still outstanding.

## Milestones and acceptance checks

| Stage | Work | Acceptance check | Status |
| --- | --- | --- | --- |
| 1. Data audit | Download pinned dataset revision; inspect columns, durations, near-full-scale peaks, empty audio, transcript labels, and original source terms. | Save counts and a review log; confirm redistribution terms before releasing any derived data. | Train/validation technical audit complete; original source rights and human label review pending |
| 2. Split protocol | Make deterministic transcript-disjoint train/validation/test manifest. | No normalized phrase crosses partitions; record clip and unique-phrase counts; freeze manifest. | Completed on 6,661 clips; see `DATA_AUDIT.md` |
| 3. Baseline | Run pretrained Whisper tiny English on validation. | Save full predictions, normalized WER/CER, term support/recall, hardware, model SHA, dataset SHA, code commit. | Full 649-clip validation run complete; clean-commit rerun pending |
| 4. Fine-tuning | Run a short smoke train, then full training on train only; select by validation WER. | No test data is read; checkpoint loads for inference; memory fits 6 GB or batch settings are adjusted and logged. | Four-train/two-validation CPU smoke and checkpoint reload complete; full CUDA run pending |
| 5. Final comparison | Freeze model, term list, normalization, and exclusions; score both checkpoints on full test once. | Same test rows, decoding settings, and metric code; report absolute and relative changes with sample counts. | Pending |
| 6. Robustness | Add reproducible white noise at fixed SNRs to audio without changing text; compare paired clean/noisy results. | Seeded transformations, documented noise method, and matched clips. Treat validation stress tests as exploratory; reserve final test until model selection. | Full 649-clip 10 dB validation stress test complete; clean-commit rerun pending |
| 7. Uncertainty and errors | Compare paired predictions with transcript-group bootstrap intervals; audit errors by term and audio quality. | Report intervals, number of unique phrases, examples, and failures; distinguish exploratory from confirmatory analyses. | 1,000-sample grouped bootstrap complete; human audio/label error review pending |
| 8. Public release | Review README, RESULTS, attribution, dependency versions, CI, and ignored files. | `git status` contains no audio, checkpoints, tokens, or private data; a new machine can follow setup. | Documentation in progress; no remote configured |

## Recommended execution order

1. Review original Kaggle/Figure Eight redistribution terms and manually sample train/validation audio and labels. Keep source data out of Git.
2. Install a CUDA-enabled PyTorch build using the official selector, verify `torch.cuda.is_available()`, then run a GPU memory smoke train.
3. Fine-tune on the complete train partition. Tune only on validation; record each change and checkpoint decision.
4. Freeze checkpoint, decoding, normalization, term list, and exclusions. Score both checkpoints on the full test once; report paired metrics and grouped uncertainty.
5. Review any failure modes and publish code plus aggregate results. Do not silently mix exploratory validation stress tests into the final test claim.

## Resume gate

Only after completed runs: describe the dataset, model, exact change in WER or term recall, and reproducibility work. If fine-tuning does not improve the baseline, report that honestly and explain the error analysis. Do not claim clinical deployment or patient safety.
