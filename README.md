# Medical Speech Lab

**Status (1 November 2026): full validation baseline and 10 dB synthetic-noise stress test measured; GPU fine-tuning and final test evaluation pending.** See [measured results](RESULTS.md).

Medical Speech Lab evaluates whether fine-tuning an English Whisper model improves transcription of short, spoken medical-symptom descriptions. The research question is narrow: **does domain adaptation improve word error rate and symptom-term recall on unseen phrases, and how does it behave under added noise?** The project is an ASR research prototype, not a clinical transcription service.

## Why this is a useful research project

It has a measurable baseline, a controlled fine-tuning experiment, a held-out test set, and an error analysis. The focus is on repeatable evidence: every result should identify the dataset and model revisions, code commit, hardware, split counts, normalization rules, and exact command. A model that performs worse after fine-tuning is still a valid finding if the experiment is sound.

The [dataset mirror](https://huggingface.co/datasets/Hani89/medical_asr_recording_dataset) describes 6,661 human-recorded English utterances, about 8.5 hours of audio, with repeated symptom phrases and some noisy or incorrect labels. It lists Apache-2.0; check the [original Kaggle source](https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent) before redistributing any data. This repository does **not** contain audio, transcripts, or model weights. See [data provenance](docs/DATA_PROVENANCE.md).

## Repository layout

```text
medspeech/                  Data protocol, metrics, evaluation, training
tests/                      Small offline checks for split and normalization logic
docs/IMPLEMENTATION_PLAN.md Milestones, acceptance checks, and remaining work
docs/METHODOLOGY.md         Experimental design and reporting rules
docs/DATA_PROVENANCE.md     Dataset provenance and limitations
docs/DATA_AUDIT.md          Measured split and audio properties
terms.txt                   Fixed, small symptom-term list for exact-match recall
RESULTS.md                  Measured validation results and remaining experiment
requirements.txt            Runtime dependencies excluding PyTorch
```

## Local setup

Use Python 3.11 or 3.12. On Windows PowerShell:

```powershell
cd medical-speech-lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Install a CUDA-enabled PyTorch build from https://pytorch.org/get-started/locally/ first.
python -c "import torch; print(torch.cuda.is_available())"  # Must print True for fine-tuning
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m medspeech.prepare --out runs/split_manifest.json
python -m medspeech.audit --manifest runs/split_manifest.json --split train --out runs/audit_train.json
python -m medspeech.audit --manifest runs/split_manifest.json --split validation --out runs/audit_validation.json
python -m medspeech.evaluate --manifest runs/split_manifest.json --split validation --model openai/whisper-tiny.en --limit 100 --out runs/baseline_smoke
python -m medspeech.train --manifest runs/split_manifest.json --out checkpoints/whisper_tiny_smoke --train-limit 100 --validation-limit 20 --epochs 1
```

If CUDA installation is unavailable, the training code can be checked on a very small CPU subset with `--allow-cpu --train-limit 4 --validation-limit 2 --epochs 1`. This is only a code-path check; it is not a meaningful fine-tuned model.

The limited runs only check the workflow. The source dataset stores audio as channel-first stereo arrays; `medspeech.audio` converts them deterministically to mono before model input. The baseline and artificial-noise stress test have now been run on **all validation clips**. Train without limits, choose the model and settings using validation only, then freeze the protocol. Finally, evaluate both pretrained and fine-tuned checkpoints on the entire test partition:

```powershell
python -m medspeech.evaluate --manifest runs/split_manifest.json --split validation --model openai/whisper-tiny.en --revision 87c7102498dcde7456f24cfd30239ca606ed9063 --out runs/baseline_validation
python -m medspeech.train --manifest runs/split_manifest.json --out checkpoints/whisper_tiny --epochs 3
$baseRevision = (Get-Content checkpoints/whisper_tiny/run_config.json -Raw | ConvertFrom-Json).base_model_revision
python -m medspeech.evaluate --manifest runs/split_manifest.json --split test --final-test --model openai/whisper-tiny.en --revision $baseRevision --out runs/baseline_test
python -m medspeech.evaluate --manifest runs/split_manifest.json --split test --final-test --model checkpoints/whisper_tiny/best_model --out runs/finetuned_test
python -m pip freeze > runs/environment.txt
```

The test command intentionally requires `--final-test` and forbids a row limit. If the model or preprocessing changes after seeing test results, document the test as exploratory and obtain a fresh held-out evaluation set before claiming a confirmatory improvement. Details and remaining milestones are in [the implementation plan](docs/IMPLEMENTATION_PLAN.md).

## Evaluation

`metrics.json` reports corpus WER, corpus CER, and exact normalized recall for terms in `terms.txt`. `predictions.jsonl` contains the references and model outputs for error analysis; it is ignored by Git. The current term list is a small, fixed subset of symptom words, so its recall is **not** a comprehensive clinical terminology metric. Consult [the methodology](docs/METHODOLOGY.md) before interpreting numbers.

## Current verification and limits

Offline unit tests, compilation, dataset preparation, train/validation audio audit, full validation inference, and a tiny CPU training/checkpoint-reload smoke check have run. The local RTX 4050 was detected, but the CUDA PyTorch wheel download stalled; full GPU fine-tuning has **not** run. The test partition remains untouched. The dataset does not include verified speaker IDs or actual clinical conversations; this benchmark cannot establish speaker-independent performance, medical safety, multilingual ASR, or diarization. Resume claims must reflect only measured work and must not describe this as a successful fine-tuning study yet.
