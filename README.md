# Medical Speech Lab

**A reproducible English medical-symptom ASR study with a bounded inference demo.** The repository fine-tunes a pinned Whisper tiny model, compares it with the pretrained baseline on a frozen transcript-disjoint test set, and measures sensitivity to synthetic noise. The web app accepts short recordings and displays the baseline transcript without storing uploads.

> **Research status — 1 November 2026:** full GPU training and the one-time held-out test comparison are complete. The demo, if deployed, serves the **pretrained baseline**; fine-tuned weights remain local. Neither model is clinically validated.

![Medical Speech Lab demo with audio upload, transcript panel, and measured validation results](docs/demo-desktop.png)

## Held-out test results

The [pinned public dataset mirror](https://huggingface.co/datasets/Hani89/medical_asr_recording_dataset) contains 6,661 short English symptom utterances. We repartitioned all source clips by normalized transcript hash so the same phrase never crosses train, validation, and test. The checkpoint was selected on 649 validation clips; the frozen protocol was committed before either model was scored on the 557-clip test partition.

| Model on 557 test clips | WER ↓ | CER ↓ | Fixed symptom-term recall ↑ |
| --- | ---: | ---: | ---: |
| Pretrained Whisper tiny English | 19.63% | 11.23% | 32/46 (69.57%) |
| Fine-tuned checkpoint | **12.60%** | **6.48%** | **41/46 (89.13%)** |

The paired WER reduction is **7.03 percentage points**; transcript-group bootstrap (59 groups, 1,000 samples, seed 42) gives a 95% percentile interval of **4.40–10.22 points**. The dataset contains short prompted phrases and no speaker IDs, so these results do not establish performance on clinical dictation or unseen speakers. Training settings, validation and noise results, exact revisions, and limitations are in [RESULTS.md](RESULTS.md).

## What is built

- **Research pipeline:** pinned dataset/model revisions; transcript-disjoint split; stereo-to-mono conversion; train/validation audio audit; Whisper fine-tuning; corpus WER/CER; exact lexical symptom-term recall; paired, phrase-group bootstrap; deterministic noise stress test.
- **Inference service:** FastAPI with one loaded model per process, one in-flight inference, per-IP request limit, bounded WAV streaming, duration and sample validation, 16 kHz resampling, readiness and liveness checks, and explicit research-only metadata.
- **Web demo:** responsive upload and microphone flow, in-browser WAV conversion, transcript review/copy, clear loading/error states, and measured benchmark context. The UI never sends audio to another origin.
- **Deployment:** Docker image with the pinned public baseline baked in; Caddy HTTPS, request-size limit, and a dedicated AWS CloudFormation stack with Systems Manager access and no SSH port.

```mermaid
flowchart LR
    A[Browser: upload or microphone] -->|WAV, ≤30 s| B[Caddy HTTPS]
    B -->|≤8 MB| C[FastAPI validation]
    C --> D[Mono 16 kHz resampling]
    D --> E[Pinned Whisper baseline]
    E -->|Transcript only| A
    F[Pinned public dataset] --> G[Transcript-disjoint manifest]
    G --> H[Train / validation / held-out test]
    H --> I[Evaluation and grouped bootstrap]
```

## Run locally

Use Python 3.11 or 3.12. Install a PyTorch build appropriate for your machine from the [official selector](https://pytorch.org/get-started/locally/). CPU is sufficient for the public demo and small smoke checks; full fine-tuning requires a working CUDA GPU.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.12.0
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
uvicorn medspeech.service:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The first model load may take time; `/health/ready` returns 200 once ready. The browser can upload audio or record through a microphone on localhost/HTTPS. API documentation is at `/docs`.

### API contract

```bash
curl -X POST http://127.0.0.1:8000/api/v1/transcribe \
  -H 'Content-Type: audio/wav' \
  --data-binary @short_recording.wav
```

The endpoint accepts a raw WAV body up to 8 MB, 30 seconds, one or two channels, and 8–48 kHz. It returns a transcript, clip duration, processing time, model identifier/revision, and research-only flag. Invalid input returns 413/415/422; concurrent work or more than 12 requests per IP per minute may receive 429. The service does not write audio or transcripts to disk or application logs. It is a low-traffic research demo without authentication or an SLA; **do not upload patient information**.

## Reproduce the study

The source dataset is fetched from Hugging Face at revision `8b2b2bd4233140705f1f7bb48411b49cd188d89c`. It is not committed here. The baseline model is pinned at `87c7102498dcde7456f24cfd30239ca606ed9063`.

```powershell
python -m medspeech.prepare --out runs/split_manifest.json
python -m medspeech.audit --manifest runs/split_manifest.json --split train --out runs/audit_train.json
python -m medspeech.audit --manifest runs/split_manifest.json --split validation --out runs/audit_validation.json
python -m medspeech.evaluate --manifest runs/split_manifest.json --split validation --model openai/whisper-tiny.en --revision 87c7102498dcde7456f24cfd30239ca606ed9063 --out runs/baseline_validation
python -m medspeech.evaluate --manifest runs/split_manifest.json --split validation --model openai/whisper-tiny.en --revision 87c7102498dcde7456f24cfd30239ca606ed9063 --snr-db 10 --noise-seed 42 --out runs/baseline_validation_10db
python -m medspeech.compare --baseline runs/baseline_validation/predictions.jsonl --candidate runs/baseline_validation_10db/predictions.jsonl --out runs/clean_vs_noise.json
```

Choose new output paths if those directories already exist. Outputs, transcripts, model weights, and Hugging Face caches are ignored by Git. The [methodology](docs/METHODOLOGY.md) defines normalization, terms, split rules, and interpretation. The [data audit](docs/DATA_AUDIT.md) records the actual channel and duration properties.

### Fine-tuning and final comparison

The full run trained on 5,455 clips for three epochs on an RTX 4050 Laptop GPU (6 GB); epoch 1 gave the lowest training-loop validation WER. The saved checkpoint was independently evaluated on validation before the [protocol freeze](docs/PROTOCOL_FREEZE.md). To reproduce, install CUDA PyTorch, confirm `torch.cuda.is_available()`, and use a fresh output directory. The tested WSL setup is in [GPU setup](scripts/setup_wsl_gpu.sh).

```powershell
python -m medspeech.train --manifest runs/split_manifest.json --out checkpoints/whisper_tiny_full --epochs 3 --batch-size 2 --gradient-accumulation 8
python -m medspeech.evaluate --manifest runs/split_manifest.json --split validation --model checkpoints/whisper_tiny_full/best_model --out runs/finetuned_validation
```

The final test command intentionally requires `--final-test` and disallows a row limit:

```powershell
python -m medspeech.evaluate --manifest runs/split_manifest.json --split test --final-test --model openai/whisper-tiny.en --revision 87c7102498dcde7456f24cfd30239ca606ed9063 --out runs/baseline_test
python -m medspeech.evaluate --manifest runs/split_manifest.json --split test --final-test --model checkpoints/whisper_tiny_full/best_model --out runs/finetuned_test
python -m medspeech.compare --baseline runs/baseline_test/predictions.jsonl --candidate runs/finetuned_test/predictions.jsonl --out runs/final_comparison.json
```

If settings change after viewing test results, treat this test as exploratory and obtain a fresh holdout for a confirmatory claim. The published aggregate metrics come from clean commit `0a9559b`; raw predictions and weights are kept out of Git.

## Deploy

See [deployment instructions](docs/DEPLOYMENT.md) for Docker Compose, AWS CloudFormation, health checks, updates, and teardown. The standard image bakes in the public pretrained baseline; fine-tuned weights are never pushed to GitHub. The CloudFormation stack incurs ongoing EC2, storage, and public IPv4 charges.

## Repository guide

| Path | Purpose |
| --- | --- |
| `medspeech/` | Data protocol, audio processing, training, evaluation, comparison, and API |
| `web/` | Same-origin browser demo |
| `tests/` | Offline protocol, audio, comparison, and API tests |
| `deploy/` | Caddy, Compose, and AWS infrastructure |
| `docs/METHODOLOGY.md` | Experimental design and claims boundary |
| `docs/DATA_PROVENANCE.md` | Source attribution and redistribution limits |
| `RESULTS.md` | Measured validation, robustness, and held-out test metrics |
| `terms.txt` | Fixed lexical symptom terms used in the benchmark |

## Limits and responsible use

The dataset contains short prompted utterances, not physician dictation or patient conversations. Speaker IDs are unavailable, so the split does not establish speaker independence. The small English term list is a lexical probe, not medical entity extraction. The source mirror carries an Apache-2.0 tag, but original Kaggle/Figure Eight audio redistribution rights remain unverified; this repository includes **no source audio or transcripts**. The model can omit or invent medically important words. This project does not establish clinical safety, multilingual ASR, diarization, or healthcare compliance. See [data provenance](docs/DATA_PROVENANCE.md) and the [implementation plan](docs/IMPLEMENTATION_PLAN.md).
