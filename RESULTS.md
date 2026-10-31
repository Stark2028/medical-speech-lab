# Results

**Status (1 November 2026):** full GPU fine-tuning, independent validation, and the one-time held-out test comparison are complete. [The final protocol was committed](docs/PROTOCOL_FREEZE.md) before either test evaluation.

## Experimental record

| Item | Recorded value |
| --- | --- |
| Dataset | `Hani89/medical_asr_recording_dataset` at `8b2b2bd4233140705f1f7bb48411b49cd188d89c` |
| Baseline | `openai/whisper-tiny.en` at `87c7102498dcde7456f24cfd30239ca606ed9063` |
| Split | 5,455 train / 649 validation / 557 test clips; 577 / 65 / 59 unique normalized phrases |
| Manifest SHA-256 | `f9e3135aa171cebf5f38baa3d392c42dcade38ff7617110fb25683f542b7f432` |
| Selected checkpoint artifact SHA-256 | `d1b972ee326516a701748eaeae710a6e356f6873434d1767f1125a1d81779427` |
| Fixed term list SHA-256 | `7b4ea73fdafa4c82cc83f63084e20c3ac33a077a3f4d7546b5200f3ad66880e6` |
| Training | RTX 4050 Laptop GPU (6 GB), PyTorch 2.12.0+cu130, 3 epochs, batch 2, accumulation 8, learning rate 1e-5, seed 42, 808.53 s |
| Model selection | Lowest training-loop validation WER: epoch 1, 11.03%; independent saved-checkpoint validation WER: 11.06% |
| Evaluation software | PyTorch 2.12.0; Transformers 4.50.3; Datasets 2.21.0; Jiwer 4.0.0 |
| Evaluation commit | `0a9559b7d9f3ecc71cb786f321ae1c574954c42c`; both test metrics recorded `code_dirty: false` |

The independent validation and test runs used identical audio conversion, text normalization, metric code, and `max_new_tokens=128`. Each model retained its saved generation configuration. Term recall is an exact lexical match for the fixed list in [terms.txt](terms.txt), counting one occurrence per clip and term. It is **not** medical entity recognition or a clinical safety metric.

## Held-out test: 557 clips, 59 transcript groups

| Model | Corpus WER ↓ | Corpus CER ↓ | Fixed symptom-term recall ↑ |
| --- | ---: | ---: | ---: |
| Pinned pretrained baseline | 19.63% | 11.23% | 32/46 (69.57%) |
| Selected fine-tuned checkpoint | **12.60%** | **6.48%** | **41/46 (89.13%)** |

The paired candidate-minus-baseline WER difference is **-7.03 percentage points**, a **35.8% relative WER reduction**. A 1,000-sample bootstrap resampling the 59 normalized-transcript groups (seed 42) gave a 95% percentile interval of **-10.22 to -4.40 percentage points**. The interval describes sampling uncertainty over phrase groups in this dataset; it does not correct for unknown speaker overlap, label issues, or the gap from prompted utterances to clinical audio. The term-recall difference is 9 additional exact matches among 46 reference occurrences; that small denominator warrants caution.

## Validation and noise stress test

| Input / model | WER ↓ | CER ↓ | Fixed term recall ↑ |
| --- | ---: | ---: | ---: |
| Clean / pinned baseline, independent GPU run | 17.04% | 9.89% | 38/68 (55.88%) |
| Clean / selected fine-tuned checkpoint | 11.06% | 6.15% | 57/68 (83.82%) |
| 10 dB seeded white Gaussian noise / pinned baseline, earlier CPU run | 32.17% | 21.49% | 20/68 (29.41%) |

On clean validation, the paired candidate-minus-baseline WER difference is **-5.98 percentage points**, with a 95% phrase-group bootstrap interval of **-7.72 to -4.45 points** (65 groups, 1,000 samples, seed 42). The earlier CPU clean baseline was 17.01% WER and 9.88% CER, so the GPU clean run differs slightly by execution environment. The noise result is paired with that earlier CPU clean run: noisy-minus-clean WER **+15.16 points**, 95% interval **+12.54 to +17.94 points**. White noise at a fixed SNR is an artificial perturbation, not real clinical noise.

## Reproduce

Install dependencies as in [README.md](README.md), or use the tested [WSL GPU setup](scripts/setup_wsl_gpu.sh). Use new output paths for reruns. The pinned dataset/model must be accessible through the Hub or local cache. The complete pipeline is:

```text
python -m medspeech.prepare --out runs/split_manifest.json
python -m medspeech.train --manifest runs/split_manifest.json --out checkpoints/whisper_tiny_full --epochs 3 --batch-size 2 --gradient-accumulation 8 --learning-rate 1e-5 --seed 42
python -m medspeech.evaluate --manifest runs/split_manifest.json --split validation --model checkpoints/whisper_tiny_full/best_model --out runs/finetuned_validation
python -m medspeech.evaluate --manifest runs/split_manifest.json --split test --final-test --model openai/whisper-tiny.en --revision 87c7102498dcde7456f24cfd30239ca606ed9063 --out runs/baseline_test
python -m medspeech.evaluate --manifest runs/split_manifest.json --split test --final-test --model checkpoints/whisper_tiny_full/best_model --out runs/finetuned_test
python -m medspeech.compare --baseline runs/baseline_test/predictions.jsonl --candidate runs/finetuned_test/predictions.jsonl --samples 1000 --seed 42 --out runs/final_comparison.json
```

The actual run outputs, raw predictions, and weights are ignored by Git. Each evaluation directory contains `predictions.jsonl` and `metrics.json`; the latter records revisions, manifest/checkpoint hashes, versions, hardware, term-list hash, code commit, and worktree state. The full commands, parameter freeze, and selected checkpoint are in [PROTOCOL_FREEZE.md](docs/PROTOCOL_FREEZE.md).

## Limits

The dataset consists of short prompted English symptom phrases, not physician dictation or patient conversations. Speaker IDs are absent; transcript-disjoint splitting does not establish speaker independence. Some validation errors included missed symptom terms and fluent but unrelated text, and the test aggregate still contains errors. Human audio/reference review was not completed, so model mismatches cannot all be attributed to recognition. The original source-audio redistribution rights remain unverified; no audio, transcripts, predictions, or model weights are published. These findings do not establish clinical readiness, multilingual ASR, or patient safety. See [data provenance](docs/DATA_PROVENANCE.md).
