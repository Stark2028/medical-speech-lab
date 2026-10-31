# Methodology

## Data split

The Hugging Face mirror has repeated transcripts. All clips whose transcript normalizes to the same text receive the same partition via SHA-256 of the normalized text (80/10/10 buckets). The source dataset revision and source-split fingerprints are saved in `runs/split_manifest.json`. The split protects against exact transcript overlap; it does **not** protect against speaker overlap because speaker identifiers are unavailable.

Do not move clips between partitions to improve metrics. Inspect the data for invalid or empty audio and document exclusions before final test scoring. Audit reference transcription errors on train and validation only; the final test must remain unseen during model selection.

## Models and tuning

The baseline is `openai/whisper-tiny.en`, a pretrained English Whisper model. The training script pins its Hub revision, trains on the train partition, and selects the best checkpoint on validation WER. The saved checkpoint is compared against the same pinned baseline with the same decoding pipeline. A 6 GB GPU is available locally; actual fit must be verified with a smoke run. Record any changed batch size, gradient accumulation, learning rate, epochs, and random seed.

## Metrics

- **Corpus WER:** word edit distance divided by total reference words. Lower is better. It can exceed 100% when insertions are numerous.
- **Corpus CER:** character edit distance divided by total reference characters.
- **Symptom-term recall:** number of occurrences of a fixed term that are also present in the normalized hypothesis divided by occurrences in the reference. This is exact lexical matching, not entity extraction; report denominator and `null` if no terms occur.
- Use the *same* lowercase/punctuation normalization for all model outputs and references. Preserve raw text in prediction artifacts for auditing.
- Report both clip count and unique normalized phrase count. Repeated phrases are correlated; use phrase-group resampling for future confidence intervals.

The fixed term list is intentionally small and is committed before test scoring. Do not add terms after inspecting test outputs and then claim the same test as confirmatory.

## Interpretation

Compare paired predictions on identical test clips. A lower WER does not guarantee better medical-term recognition or clinical safety. Inspect substitutions that change symptom meaning. Report model failures, dataset label issues, and noisy-recording behavior. No patient data is included or required.
