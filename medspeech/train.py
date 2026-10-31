"""Fine-tune Whisper tiny on transcript-disjoint train/validation partitions.

Never read the test partition during training or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import mono_16k
from .protocol import normalize_text


@dataclass
class SpeechCollator:
    processor: object

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        features = self.processor.feature_extractor.pad(
            [{"input_features": item["input_features"],
              "attention_mask": item["attention_mask"]} for item in batch],
            return_tensors="pt")
        labels = self.processor.tokenizer.pad(
            [{"input_ids": item["labels"]} for item in batch],
            return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        if (ids[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            ids = ids[:, 1:]
        features["labels"] = ids
        return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="openai/whisper-tiny.en")
    parser.add_argument("--revision", help="Pinned Hub model revision; omit for a local model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Only for tiny code-path smoke checks without CUDA")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation < 1
            or args.learning_rate <= 0 or any(
                value is not None and value < 1
                for value in (args.train_limit, args.validation_limit))):
        parser.error("Epochs, batch size, accumulation, learning rate, and limits must be positive")
    if args.out.exists() and any(args.out.iterdir()) and not args.resume_from_checkpoint:
        parser.error("Output directory is not empty; choose a new run directory")
    if args.resume_from_checkpoint and not args.resume_from_checkpoint.is_dir():
        parser.error("--resume-from-checkpoint must be an existing checkpoint directory")

    import torch
    from datasets import load_dataset
    from jiwer import wer
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor)

    has_cuda = torch.cuda.is_available()
    if not has_cuda and not args.allow_cpu:
        raise RuntimeError("Full fine-tuning requires CUDA; use --allow-cpu only for a tiny smoke check")
    if args.allow_cpu and (args.train_limit is None or args.train_limit > 16 or
                           args.validation_limit is None or args.validation_limit > 8):
        parser.error("--allow-cpu requires --train-limit <= 16 and --validation-limit <= 8")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    raw = load_dataset(manifest["dataset_id"], revision=manifest["dataset_revision"])
    actual = {name: subset._fingerprint for name, subset in raw.items()}
    if actual != manifest["source_fingerprints"]:
        raise RuntimeError("Dataset changed since manifest creation; regenerate and rerun all experiments")
    model_revision = None if Path(args.model).exists() else (
        args.revision or ("87c7102498dcde7456f24cfd30239ca606ed9063"
                          if args.model == "openai/whisper-tiny.en" else None))
    if model_revision is None and not Path(args.model).exists():
        from huggingface_hub import HfApi
        model_revision = HfApi().model_info(args.model).sha
    processor = WhisperProcessor.from_pretrained(args.model, revision=model_revision)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, revision=model_revision)
    model.config.use_cache = False
    model.generation_config.forced_decoder_ids = None

    from torch.utils.data import Dataset

    class FeatureDataset(Dataset):
        def __init__(self, split: str, limit: int | None):
            self.rows = [r for r in manifest["rows"] if r["partition"] == split]
            if limit is not None:
                self.rows = self.rows[:limit]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict:
            row = self.rows[index]
            item = raw[row["source_split"]][row["index"]]
            audio = item["audio"]
            encoded = processor.feature_extractor(
                mono_16k(audio), sampling_rate=16000,
                return_attention_mask=True)
            return {
                "input_features": encoded.input_features[0],
                "attention_mask": encoded.attention_mask[0],
                "labels": processor.tokenizer(item["sentence"]).input_ids,
            }

    train_data = FeatureDataset("train", args.train_limit)
    validation_data = FeatureDataset("validation", args.validation_limit)
    if not train_data or not validation_data:
        raise ValueError("Train and validation partitions must both contain data")

    def compute_metrics(prediction) -> dict:
        pred_ids = prediction.predictions
        label_ids = np.where(prediction.label_ids == -100,
                             processor.tokenizer.pad_token_id, prediction.label_ids)
        preds = [normalize_text(x) for x in processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True)]
        refs = [normalize_text(x) for x in processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True)]
        return {"wer": wer(refs, preds)}

    args.out.mkdir(parents=True, exist_ok=True)
    training = Seq2SeqTrainingArguments(
        output_dir=str(args.out), per_device_train_batch_size=args.batch_size if has_cuda else 1,
        per_device_eval_batch_size=args.batch_size if has_cuda else 1,
        gradient_accumulation_steps=args.gradient_accumulation if has_cuda else 1,
        learning_rate=args.learning_rate, warmup_steps=20 if has_cuda else 0,
        num_train_epochs=args.epochs,
        eval_strategy="epoch", save_strategy="epoch", logging_steps=25,
        predict_with_generate=True, generation_max_length=128,
        fp16=has_cuda, gradient_checkpointing=has_cuda,
        load_best_model_at_end=True,
        metric_for_best_model="wer", greater_is_better=False,
        save_total_limit=2, report_to=[], seed=args.seed,
        dataloader_pin_memory=has_cuda)
    trainer = Seq2SeqTrainer(
        model=model, args=training, train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=SpeechCollator(processor), compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor)
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint)
                  if args.resume_from_checkpoint else None)
    trainer.save_model(str(args.out / "best_model"))
    processor.save_pretrained(str(args.out / "best_model"))
    git_root = Path(__file__).resolve().parents[1]
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=git_root,
                                capture_output=True, text=True, check=False)
    git_status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                                cwd=git_root, capture_output=True, text=True, check=False)
    (args.out / "run_config.json").write_text(json.dumps({
        "dataset": manifest["dataset_id"],
        "dataset_revision": manifest["dataset_revision"],
        "base_model": args.model, "base_model_revision": model_revision,
        "train_count": len(train_data), "validation_count": len(validation_data),
        "epochs": args.epochs,
        "batch_size": args.batch_size if has_cuda else 1,
        "gradient_accumulation": args.gradient_accumulation if has_cuda else 1,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "best_validation_wer": trainer.state.best_metric,
        "torch_version": torch.__version__,
        "code_commit": git_commit.stdout.strip() if git_commit.returncode == 0 else None,
        "code_dirty": bool(git_status.stdout.strip()) if git_status.returncode == 0 else None,
        "device": torch.cuda.get_device_name(0) if has_cuda else "CPU",
        "smoke_run": args.train_limit is not None or args.validation_limit is not None,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
