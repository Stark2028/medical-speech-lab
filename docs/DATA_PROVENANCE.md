# Data provenance and release rules

## Source

- Hugging Face dataset: [`Hani89/medical_asr_recording_dataset`](https://huggingface.co/datasets/Hani89/medical_asr_recording_dataset).
- The dataset card attributes its audio and transcripts to [Medical Speech, Transcription, and Intent on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent).
- The card reports 6,661 rows, about 8.5 hours of human-recorded English symptom utterances, 704 distinct sentence values in the dataset viewer, and an Apache-2.0 license tag. It explicitly warns about bad labels and poor audio.

The Hugging Face tag is not, by itself, proof that the original audio can be redistributed under that license. **Before public release, verify the original Kaggle/Figure Eight terms and record the result here.** This repo references the dataset but does not mirror any audio or transcripts.

## Known limits

- Symptom prompts are not clinical consultations or medical records.
- Speaker identifiers and demographic metadata are not verified, so speaker-disjoint or fairness claims are unsupported.
- The dataset is English only; no multilingual, diarization, or long-form ASR claim can be made.
- Transcript labels may contain mistakes; use a documented audit and avoid changing held-out references based on model predictions.

## Git hygiene

Do not commit downloaded data, cached models, checkpoints, generated predictions, access tokens, or private recordings. The `.gitignore` excludes local artifact directories. Review `git status` and large files before any push.
