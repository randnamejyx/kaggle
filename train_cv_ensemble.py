"""Train a five-fold text-only NB-SVM ensemble with an OOF threshold."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

from train_optimized import (
    SEED,
    best_threshold,
    fit_features,
    load_csv,
    nb_ratio,
    transform_features,
)


def train(args):
    train_data = load_csv(args.train, [args.text_column, "label"])
    test_data = load_csv(args.test, [args.text_column, "id"])
    labels = train_data["label"].str.strip()
    if set(labels) != {"0", "1"}:
        raise ValueError("training labels must contain only both 0 and 1")
    labels = labels.astype(int).to_numpy()
    if test_data["id"].str.strip().eq("").any() or test_data["id"].duplicated().any():
        raise ValueError("test IDs must be nonempty and unique")

    args.output.mkdir(parents=True, exist_ok=True)
    texts = train_data[args.text_column]
    test_texts = test_data[args.text_column]
    groups = texts.str.split().str.join(" ").str.casefold()
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=SEED
    )
    oof_scores = np.zeros(len(train_data), dtype=float)
    test_scores = np.zeros(len(test_data), dtype=float)
    fold_results = []

    for fold, (fit_indices, valid_indices) in enumerate(
        splitter.split(texts, labels, groups=groups), start=1
    ):
        print(f"Fold {fold}/{args.folds}: fitting TF-IDF...", flush=True)
        word, char, fit_matrix = fit_features(texts.iloc[fit_indices])
        valid_matrix = transform_features(word, char, texts.iloc[valid_indices])
        fold_test_matrix = transform_features(word, char, test_texts)
        fit_labels = labels[fit_indices]
        ratio = nb_ratio(fit_matrix, fit_labels)
        classifier = LinearSVC(C=args.c, max_iter=5000, random_state=SEED)
        classifier.fit(fit_matrix.multiply(ratio), fit_labels)
        validation_scores = classifier.decision_function(valid_matrix.multiply(ratio))
        oof_scores[valid_indices] = validation_scores
        test_scores += classifier.decision_function(
            fold_test_matrix.multiply(ratio)
        ) / args.folds
        default_f1 = f1_score(
            labels[valid_indices], validation_scores >= 0, zero_division=0
        )
        result = {
            "fold": fold,
            "fit_rows": len(fit_indices),
            "validation_rows": len(valid_indices),
            "f1_default": default_f1,
            "features": fit_matrix.shape[1],
        }
        fold_results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        joblib.dump(
            {
                "word_vectorizer": word,
                "char_vectorizer": char,
                "classifier": classifier,
                "nb_ratio": ratio,
                "text_column": args.text_column,
            },
            args.output / f"model_fold_{fold}.joblib",
            compress=3,
        )

    threshold, oof_f1 = best_threshold(labels, oof_scores)
    oof_predictions = (oof_scores >= threshold).astype(int)
    test_predictions = (test_scores >= threshold).astype(int)
    submission = pd.DataFrame({"id": test_data["id"], "label": test_predictions})
    submission.to_csv(args.output / "submission.csv", index=False)
    np.save(args.output / "oof_scores.npy", oof_scores)
    np.save(args.output / "test_scores.npy", test_scores)
    metrics = {
        "seed": SEED,
        "method": "five_fold_text_only_tfidf_nb_svm_ensemble",
        "uses_id_as_feature": False,
        "folds": args.folds,
        "C": args.c,
        "threshold": threshold,
        "oof_f1": oof_f1,
        "oof_precision": precision_score(labels, oof_predictions, zero_division=0),
        "oof_recall": recall_score(labels, oof_predictions, zero_division=0),
        "fold_results": fold_results,
        "test_label_counts": {
            str(key): int(value)
            for key, value in submission["label"].value_counts().sort_index().items()
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Results saved in {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/optimized_cv"))
    args = parser.parse_args()
    try:
        train(args)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Training stopped: {error}\n")


if __name__ == "__main__":
    main()
