# Final test protocol freeze — 1 November 2026

This record was committed before either model was evaluated on the 557-clip test partition. Training and model selection used only the 5,455 training and 649 validation clips. The test partition was not used to choose the checkpoint, preprocessing, normalization, terms, or decoding settings.

| Item | Frozen value |
| --- | --- |
| Source dataset | `Hani89/medical_asr_recording_dataset`, revision `8b2b2bd4233140705f1f7bb48411b49cd188d89c` |
| Split manifest SHA-256 | `f9e3135aa171cebf5f38baa3d392c42dcade38ff7617110fb25683f542b7f432` |
| Baseline | `openai/whisper-tiny.en`, revision `87c7102498dcde7456f24cfd30239ca606ed9063` |
| Selected candidate | `checkpoints/wsl_full_3e/best_model`, local artifact SHA-256 `d1b972ee326516a701748eaeae710a6e356f6873434d1767f1125a1d81779427` |
| Checkpoint selection | Lowest training-loop validation WER; epoch 1 of 3, 11.03% |
| Independent validation | Baseline WER 17.04%; candidate WER 11.06%; 649 clips and 65 transcript groups |
| Terms file SHA-256 | `7b4ea73fdafa4c82cc83f63084e20c3ac33a077a3f4d7546b5200f3ad66880e6` |
| Audio | Mean stereo channels to mono; resample to 16 kHz; no noise added |
| Decoding | Hugging Face Transformers 4.50.3 `model.generate` with `max_new_tokens=128`, model generation configuration otherwise loaded from the pinned baseline or saved candidate |
| Text normalization | NFKC, lowercase, punctuation to spaces, collapse whitespace |
| Metrics | Corpus WER, corpus CER, exact lexical symptom-term recall; paired WER delta and transcript-group bootstrap, 1,000 samples, seed 42 |
| Exclusions | None; score every clip in the 557-clip test partition |

The candidate was trained for three epochs with batch size 2, gradient accumulation 8, learning rate 1e-5, seed 42, on an RTX 4050 Laptop GPU (6 GB). The complete training run selected epoch 1 by validation WER. The independently loaded saved candidate scored 11.06% WER on the same validation partition; the small difference from training-loop WER reflects separate generation and metric paths, not a new selection decision.

The test will be evaluated once for each model with the commands in `README.md`, with `--final-test` and no row limit. If later analysis motivates changes, this test becomes exploratory and a new holdout is required for a confirmatory claim. No patient audio, private recordings, source clips, predictions, or model weights will be published.
