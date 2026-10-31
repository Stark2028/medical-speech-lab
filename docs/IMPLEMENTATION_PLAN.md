# Implementation plan

## Objective

Answer whether fine-tuning Whisper tiny English on short medical-symptom speech improves transcription of unseen phrases. Report overall ASR quality and exact symptom-term recall, with honest limitations. The completed held-out comparison is in [RESULTS.md](../RESULTS.md).

## Milestones and acceptance checks

| Stage | Work | Acceptance check | Status |
| --- | --- | --- | --- |
| 1. Data audit | Download pinned dataset revision; inspect columns, durations, near-full-scale peaks, empty audio, transcript labels, and original source terms. | Save counts and a review log; confirm redistribution terms before releasing any derived data. | Train/validation technical audit complete; original source rights and human label review pending |
| 2. Split protocol | Make deterministic transcript-disjoint train/validation/test manifest. | No normalized phrase crosses partitions; record clip and unique-phrase counts; freeze manifest. | Completed on 6,661 clips; see `DATA_AUDIT.md` |
| 3. Baseline | Run pretrained Whisper tiny English on validation. | Save full predictions, normalized WER/CER, term support/recall, hardware, model SHA, dataset SHA, code commit. | Full 649-clip validation run reproduced from clean commit `5064084` |
| 4. Fine-tuning | Run a short smoke train, then full training on train only; select by validation WER. | No test data is read; checkpoint loads for inference; memory fits 6 GB or batch settings are adjusted and logged. | Completed on 5,455 train clips, RTX 4050 GPU; epoch 1 checkpoint selected |
| 5. Final comparison | Freeze model, term list, normalization, and exclusions; score both checkpoints on full test once. | Same test rows, decoding settings, and metric code; report absolute and relative changes with sample counts. | Completed on 557 clips after [protocol freeze](PROTOCOL_FREEZE.md); 19.63% to 12.60% WER |
| 6. Robustness | Add reproducible white noise at fixed SNRs to audio without changing text; compare paired clean/noisy results. | Seeded transformations, documented noise method, and matched clips. Treat validation stress tests as exploratory; reserve final test until model selection. | Full 649-clip 10 dB validation stress test reproduced from clean commit `5064084` |
| 7. Uncertainty and errors | Compare paired predictions with transcript-group bootstrap intervals; audit errors by term and audio quality. | Report intervals, number of unique phrases, examples, and failures; distinguish exploratory from confirmatory analyses. | 1,000-sample grouped test bootstrap completed; human audio/label review remains open |
| 8. Public release | Review README, RESULTS, attribution, dependency versions, CI, and ignored files. | `git status` contains no audio, checkpoints, tokens, or private data; a new machine can follow setup. | Public GitHub repo created; code and aggregate report ready for CI and deployment verification |

## Remaining research work

1. Review original Kaggle/Figure Eight redistribution terms and manually sample train/validation audio and labels. Keep source data out of Git.
2. Obtain a speaker-identified, independently collected evaluation set before claiming generalization to unseen speakers or clinical conversations.
3. Validate the service with non-sensitive public audio under realistic traffic and monitor latency, memory, and failure rates before increasing exposure.

## Resume gate

The completed run supports a resume claim of fine-tuning Whisper tiny on 5,455 training utterances and reducing held-out WER from 19.63% to 12.60% on 557 prompted clips, with a transcript-disjoint protocol and paired uncertainty estimate. Do not claim clinical deployment or patient safety.
