"""Run a Whisper checkpoint on a fixed partition and save predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

from .metrics import score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--final-test", action="store_true",
                        help="Confirm model selection is frozen before the untouched test set is scored")
    parser.add_argument("--model", default="openai/whisper-tiny.en")
    parser.add_argument("--revision", help="Pinned Hub model commit SHA; omit for local checkpoints")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--terms", type=Path, default=Path("terms.txt"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.split == "test" and (not args.final_test or args.limit is not None):
        parser.error("Test evaluation requires --final-test and does not allow --limit")
    if args.final_test and args.split != "test":
        parser.error("--final-test requires --split test")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory is not empty; choose a new run directory")

    import torch
    from datasets import Audio, load_dataset
    from huggingface_hub import HfApi
    from transformers import pipeline

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [r for r in manifest["rows"] if r["partition"] == args.split]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No rows in selected partition")
    dataset = load_dataset(manifest["dataset_id"], revision=manifest["dataset_revision"])
    actual = {name: subset._fingerprint for name, subset in dataset.items()}
    if actual != manifest["source_fingerprints"]:
        raise RuntimeError("Dataset changed since manifest creation; regenerate and rerun all experiments")
    dataset = {name: subset.cast_column("audio", Audio(sampling_rate=16000))
               for name, subset in dataset.items()}
    device = 0 if torch.cuda.is_available() else -1
    model_revision = None
    if Path(args.model).exists():
        if args.revision:
            parser.error("--revision applies only to Hub models")
    else:
        model_revision = args.revision or HfApi().model_info(args.model).sha
    recognizer = pipeline("automatic-speech-recognition", model=args.model,
                          revision=model_revision, device=device, chunk_length_s=30)
    args.out.mkdir(parents=True, exist_ok=True)
    references, predictions = [], []
    start = time.perf_counter()
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as output:
        for row in rows:
            item = dataset[row["source_split"]][row["index"]]
            audio = item["audio"]
            prediction = recognizer({"array": audio["array"],
                                     "sampling_rate": audio["sampling_rate"]})["text"]
            reference = item["sentence"]
            references.append(reference)
            predictions.append(prediction)
            record = {"source_split": row["source_split"], "index": row["index"],
                      "reference": reference, "prediction": prediction}
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    terms = [line.strip() for line in args.terms.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    metrics = score(references, predictions, terms)
    metrics.update({"model": args.model, "model_revision": model_revision,
                    "dataset": manifest["dataset_id"],
                    "dataset_revision": manifest["dataset_revision"],
                    "split": args.split, "device": torch.cuda.get_device_name(0) if device == 0 else "CPU",
                    "elapsed_seconds": round(time.perf_counter() - start, 2),
                    "limited_run": args.limit is not None,
                    "term_list_sha256": hashlib.sha256(args.terms.read_bytes()).hexdigest(),
                    "normalization": "NFKC lowercase punctuation-to-space whitespace-collapse",
                    "versions": {name: version(name) for name in
                                 ("torch", "transformers", "datasets", "jiwer")}})
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=False)
    metrics["code_commit"] = git_commit.stdout.strip() if git_commit.returncode == 0 else None
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
