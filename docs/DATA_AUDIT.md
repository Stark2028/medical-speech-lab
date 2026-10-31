# Data audit (1 November 2026)

Dataset revision: `8b2b2bd4233140705f1f7bb48411b49cd188d89c`. The original Hugging Face source contains 5,328 `train` rows and 1,333 `test` rows. For this experiment, all 6,661 clips were repartitioned by normalized transcript into 5,455 train, 649 validation, and 557 test clips. These partitions contain 577, 65, and 59 unique normalized phrases respectively (701 total).

The train and validation audio was audited before model selection. The test audio and labels remain reserved for final evaluation.

| Check | Train | Validation |
| --- | ---: | ---: |
| Clips | 5,455 | 649 |
| Valid 16 kHz audio | 5,455 | 649 |
| Two-channel arrays | 5,455 | 649 |
| Empty or non-finite audio | 0 | 0 |
| Peak amplitude at least 0.999 | 491 | 60 |
| Minimum duration | 0.683 s | 1.280 s |
| Median duration | 4.181 s | 3.994 s |
| Maximum duration | 18.483 s | 11.861 s |
| Total duration | 6.992 h | 0.766 h |

All inspected audio arrays are stored as `[channel][sample]`, despite the dataset card's 1D description. The preprocessing function averages the two channels to produce 16 kHz mono input for Whisper. A peak near full scale is **not** proof of clipping; waveform and listening checks would be needed before labeling those recordings as clipped.

The machine-generated reports are in ignored `runs/audit_train.json` and `runs/audit_validation.json`. This summary excludes audio and transcript text, so it is safe to keep in the public repository.
