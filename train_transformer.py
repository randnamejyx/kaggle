"""Fine-tune a pretrained text classifier for this competition."""

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup


def read_csv(path, labeled):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"id", "text"} | ({"label"} if labeled else set())
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path}: expected columns {sorted(required)}")
    return rows


def stratified_split(labels, validation_fraction, seed):
    generator = np.random.default_rng(seed)
    fit, validation = [], []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        generator.shuffle(indices)
        count = int(round(len(indices) * validation_fraction))
        validation.extend(indices[:count])
        fit.extend(indices[count:])
    generator.shuffle(fit)
    generator.shuffle(validation)
    return np.asarray(fit), np.asarray(validation)


def make_dataset(encodings, indices, labels=None):
    tensors = [encodings["input_ids"][indices], encodings["attention_mask"][indices]]
    if labels is not None:
        tensors.append(torch.as_tensor(labels[indices], dtype=torch.long))
    return TensorDataset(*tensors)


def score_model(model, loader, device, use_amp):
    model.eval()
    scores = []
    with torch.inference_mode():
        for batch in loader:
            input_ids, attention_mask = (
                item.to(device, non_blocking=True) for item in batch[:2]
            )
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(scores)


def best_f1(labels, scores):
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    predicted_positives = np.arange(1, len(labels) + 1)
    positives = int(labels.sum())
    f1 = 2 * true_positives / np.maximum(predicted_positives + positives, 1)
    index = int(np.argmax(f1))
    threshold = float(scores[order[index]])
    predictions = scores >= threshold
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    return {
        "threshold": threshold,
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
    }


def train_epoch(
    model, loader, optimizer, scheduler, positive_weight, device, epoch, scaler
):
    model.train()
    total_loss = 0.0
    weights = torch.tensor([1.0, positive_weight], device=device)
    for step, batch in enumerate(loader, start=1):
        input_ids, attention_mask, labels = (
            item.to(device, non_blocking=True) for item in batch
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += float(loss.detach())
        if step % 50 == 0 or step == len(loader):
            print(
                f"epoch={epoch} step={step}/{len(loader)} "
                f"loss={total_loss / step:.5f}",
                flush=True,
            )
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--model", default="microsoft/MiniLM-L12-H384-uncased")
    parser.add_argument("--output", type=Path, default=Path("artifacts/optimized_transformer"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--positive-weight", type=float, default=1.5)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"Training device: {torch.cuda.get_device_name(0)} (CUDA + FP16)", flush=True)
    else:
        print("Training device: CPU", flush=True)

    train_rows = read_csv(args.train, labeled=True)
    test_rows = read_csv(args.test, labeled=False)
    train_texts = [row["text"] for row in train_rows]
    test_texts = [row["text"] for row in test_rows]
    labels = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64)
    fit_indices, validation_indices = stratified_split(labels, 0.2, args.seed)

    print(f"Loading {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    print("Tokenizing text...", flush=True)
    train_encodings = tokenizer(
        train_texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    test_encodings = tokenizer(
        test_texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    fit_loader = DataLoader(
        make_dataset(train_encodings, fit_indices, labels),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_amp,
    )
    validation_loader = DataLoader(
        make_dataset(train_encodings, validation_indices),
        batch_size=args.batch_size * 2,
        pin_memory=use_amp,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(fit_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps,
    )
    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history = []
    best = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model,
            fit_loader,
            optimizer,
            scheduler,
            args.positive_weight,
            device,
            epoch,
            scaler,
        )
        validation_scores = score_model(model, validation_loader, device, use_amp)
        result = best_f1(labels[validation_indices], validation_scores)
        result.update({"epoch": epoch, "loss": loss})
        history.append(result)
        print("validation=" + json.dumps(result, sort_keys=True), flush=True)
        if best is None or result["f1"] > best["f1"]:
            best = result
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print("Selected=" + json.dumps(best, sort_keys=True), flush=True)
    print("Continuing the selected model for one epoch on all training rows...", flush=True)
    full_loader = DataLoader(
        make_dataset(train_encodings, np.arange(len(labels)), labels),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_amp,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate / 2)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(full_loader)
    )
    train_epoch(
        model,
        full_loader,
        optimizer,
        scheduler,
        args.positive_weight,
        device,
        "full",
        scaler,
    )
    test_loader = DataLoader(
        make_dataset(test_encodings, np.arange(len(test_rows))),
        batch_size=args.batch_size * 2,
        pin_memory=use_amp,
    )
    test_scores = score_model(model, test_loader, device, use_amp)
    predictions = (test_scores >= best["threshold"]).astype(int)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "submission.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "label"])
        writer.writerows((row["id"], int(label)) for row, label in zip(test_rows, predictions))
    np.save(args.output / "test_scores.npy", test_scores)
    metrics = {
        "seed": args.seed,
        "method": "pretrained_transformer_supervised_text_classifier",
        "model": args.model,
        "uses_id_as_feature": False,
        "best_validation": best,
        "history": history,
        "test_label_counts": {
            str(label): int((predictions == label).sum()) for label in (0, 1)
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    model.save_pretrained(args.output / "model")
    tokenizer.save_pretrained(args.output / "model")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Results saved in {args.output}", flush=True)


if __name__ == "__main__":
    main()
