# Results

**Status (1 November 2026): validation baseline and synthetic-noise stress test complete. Full GPU fine-tuning and final test comparison remain pending.** These are exploratory validation results, not held-out test claims.

| Item | Recorded value |
| --- | --- |
| Dataset | `Hani89/medical_asr_recording_dataset` at `8b2b2bd4233140705f1f7bb48411b49cd188d89c` |
| Baseline model | `openai/whisper-tiny.en` at `87c7102498dcde7456f24cfd30239ca606ed9063` |
| Train / validation / test clips | 5,455 / 649 / 557 |
| Train / validation / test unique normalized phrases | 577 / 65 / 59 |
| Validation hardware | CPU; NVIDIA RTX 4050 available locally but CUDA PyTorch was not installed for these runs |
| Software | Python 3.12.10; PyTorch 2.15.0 CPU; Transformers 4.50.3; Datasets 2.21.0; Jiwer 4.0.0 |
| Term list | [`terms.txt`](terms.txt), SHA-256 `7b4ea73fdafa4c82cc83f63084e20c3ac33a077a3f4d7546b5200f3ad66880e6` |
| Code commit | `5064084f26ec553d205b36175508de3e7347a411`; both validation evaluations recorded `code_dirty: false` |

See [data audit](docs/DATA_AUDIT.md) for measured channel, duration, and validity checks on train and validation audio. The original Kaggle audio redistribution terms remain unverified; no source audio, transcripts, predictions, or checkpoints are included in this repository.

## Validation benchmark

Both runs used the same 649 validation clips and fixed decoding (`max_new_tokens=128`). Metrics use the normalization in [methodology](docs/METHODOLOGY.md). Term recall counts one match per clip and term; it is a small lexical probe, not clinical entity recall.

| Input / model | WER ↓ | CER ↓ | Symptom-term recall ↑ |
| --- | ---: | ---: | ---: |
| Clean / pretrained Whisper tiny English | 17.01% | 9.88% | 38/68 (55.88%) |
| Seeded white Gaussian noise at 10 dB SNR / same model | 32.17% | 21.49% | 20/68 (29.41%) |

The paired noisy-minus-clean WER difference is **+15.16 percentage points**; a 1,000-sample bootstrap resampling 65 normalized-transcript groups gave a **95% percentile interval of +12.54 to +17.94 points** (seed 42). This quantifies sensitivity to this artificial perturbation on validation. It does not measure performance in a clinic or under real background noise.

## Reproduction

With the pinned dataset and model available in the local Hugging Face cache, run:

```powershell
python -m medspeech.prepare --out runs/split_manifest.json
python -m medspeech.evaluate --manifest runs/split_manifest.json --split validation --model openai/whisper-tiny.en --revision 87c7102498dcde7456f24cfd30239ca606ed9063 --out runs/baseline_validation_clean_commit
python -m medspeech.evaluate --manifest runs/split_manifest.json --split validation --model openai/whisper-tiny.en --revision 87c7102498dcde7456f24cfd30239ca606ed9063 --snr-db 10 --noise-seed 42 --out runs/baseline_validation_10db_clean_commit
python -m medspeech.compare --baseline runs/baseline_validation_clean_commit/predictions.jsonl --candidate runs/baseline_validation_10db_clean_commit/predictions.jsonl --samples 1000 --seed 42 --out runs/clean_vs_10db_validation_clean_commit.json
```

Each evaluation directory is ignored by Git and contains `predictions.jsonl` and `metrics.json`. `metrics.json` records model and dataset revisions, software versions, hardware, elapsed time, noise settings, term-list hash, commit, and whether the worktree was dirty.

## Remaining experiment

The CPU smoke train used four training clips and two validation clips for one epoch, and its saved checkpoint reloaded for inference. Those scores are not meaningful. Full training needs a CUDA-enabled PyTorch installation and a measured GPU run. After selecting the checkpoint and freezing settings on validation, score the pretrained and fine-tuned models once on the untouched 557-clip test partition. Until then, there is **no evidence that domain fine-tuning improves this dataset**.

## Error review

The clean validation predictions include missed symptom terms and occasional fluent but unrelated output on short clips. Example categories include confusion between breathing and ordinary phrases and generic phrases emitted in place of symptom descriptions. These observations come from validation predictions only; reference quality and audio intelligibility need human review before attributing every mismatch to the model.
