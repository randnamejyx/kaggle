"""Bind positional prediction arrays to their samples and cross-validation splits."""

import hashlib
import json

import numpy as np


def _digest(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def score_cache_signature(train_data, test_data, text_column, splits):
    train_columns = list(dict.fromkeys(
        (["id"] if "id" in train_data.columns else []) + [text_column, "label"]
    ))
    test_columns = list(dict.fromkeys(["id", text_column]))
    return {
        "version": 1,
        "text_column": text_column,
        "train_rows": len(train_data),
        "test_rows": len(test_data),
        "train_fingerprint": _digest([
            train_columns, *train_data[train_columns].itertuples(index=False, name=None)
        ]),
        "test_fingerprint": _digest([
            test_columns, *test_data[test_columns].itertuples(index=False, name=None)
        ]),
        "split_fingerprint": _digest(
            [fit.tolist(), valid.tolist()] for fit, valid in splits
        ),
    }


def save_score_cache(directory, oof_scores, test_scores, signature):
    # Write provenance last so incomplete runs cannot advertise a complete cache.
    manifest = directory / "score_cache.json"
    manifest.unlink(missing_ok=True)
    np.save(directory / "oof_scores.npy", oof_scores)
    np.save(directory / "test_scores.npy", test_scores)
    manifest.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")


def load_score_cache(directory, expected_signature):
    manifest = directory / "score_cache.json"
    if not manifest.is_file():
        raise ValueError(f"{directory}: prediction cache has no provenance; rerun its training script")
    signature = json.loads(manifest.read_text(encoding="utf-8"))
    if signature != expected_signature:
        raise ValueError(
            f"{directory}: prediction cache does not match current samples, row order, "
            "text column or folds; rerun its training script"
        )
    oof_scores = np.load(directory / "oof_scores.npy", allow_pickle=False)
    test_scores = np.load(directory / "test_scores.npy", allow_pickle=False)
    for name, scores, rows in (
        ("OOF", oof_scores, signature["train_rows"]),
        ("test", test_scores, signature["test_rows"]),
    ):
        if scores.shape != (rows,) or not np.isfinite(scores).all():
            raise ValueError(f"{directory}: invalid {name} prediction cache; rerun its training script")
    return oof_scores, test_scores
