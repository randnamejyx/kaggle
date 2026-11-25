"""Blend diverse five-fold text-only SVM models using OOF predictions."""

import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

from train_optimized import SEED, best_threshold, load_csv, nb_ratio


SPECS = (
    ("word_nb_c025", "word", True, 0.25),
    ("word_nb_c05", "word", True, 0.5),
    ("word_nb_c10", "word", True, 1.0),
    ("char_nb_c025", "char", True, 0.25),
    ("char_nb_c05", "char", True, 0.5),
    ("char_nb_c10", "char", True, 1.0),
    ("word_svm_c05", "word", False, 0.5),
    ("char_svm_c05", "char", False, 0.5),
    ("combined_svm_c05", "combined", False, 0.5),
)


def score_model(matrix, valid_matrix, test_matrix, labels, use_nb, c_value):
    ratio = nb_ratio(matrix, labels) if use_nb else None
    if ratio is not None:
        matrix = matrix.multiply(ratio)
        valid_matrix = valid_matrix.multiply(ratio)
        test_matrix = test_matrix.multiply(ratio)
    classifier = LinearSVC(C=c_value, max_iter=5000, random_state=SEED)
    classifier.fit(matrix, labels)
    return (
        classifier.decision_function(valid_matrix),
        classifier.decision_function(test_matrix),
    )


def metric(truth, scores):
    threshold, score = best_threshold(truth, scores)
    predictions = scores >= threshold
    return {
        "threshold": threshold,
        "f1": score,
        "precision": precision_score(truth, predictions, zero_division=0),
        "recall": recall_score(truth, predictions, zero_division=0),
    }


def normalized(scores, mean, standard_deviation):
    return (scores - mean) / standard_deviation


def search_blends(labels, oof, test):
    normalized_oof = {}
    normalized_test = {}
    for name, values in oof.items():
        mean = float(values.mean())
        deviation = float(values.std())
        normalized_oof[name] = normalized(values, mean, deviation)
        normalized_test[name] = normalized(test[name], mean, deviation)

    individual = {name: metric(labels, values) for name, values in oof.items()}
    ranked = sorted(individual, key=lambda name: individual[name]["f1"], reverse=True)
    best = {
        "names": [ranked[0]],
        "weights": [1.0],
        **metric(labels, normalized_oof[ranked[0]]),
    }

    top = ranked[:6]
    for first, second in itertools.combinations(top, 2):
        for first_weight in np.linspace(0.1, 0.9, 9):
            weights = [float(first_weight), float(1 - first_weight)]
            values = (
                weights[0] * normalized_oof[first]
                + weights[1] * normalized_oof[second]
            )
            result = metric(labels, values)
            if result["f1"] > best["f1"]:
                best = {
                    "names": [first, second],
                    "weights": weights,
                    **result,
                }

    for names in itertools.combinations(top[:5], 3):
        for first_weight in np.arange(0.1, 0.9, 0.1):
            for second_weight in np.arange(0.1, 1.0 - first_weight, 0.1):
                third_weight = 1.0 - first_weight - second_weight
                if third_weight < 0.1:
                    continue
                weights = [
                    float(first_weight),
                    float(second_weight),
                    float(third_weight),
                ]
                values = sum(
                    weight * normalized_oof[name]
                    for name, weight in zip(names, weights)
                )
                result = metric(labels, values)
                if result["f1"] > best["f1"]:
                    best = {"names": list(names), "weights": weights, **result}

    blended_oof = sum(
        weight * normalized_oof[name]
        for name, weight in zip(best["names"], best["weights"])
    )
    blended_test = sum(
        weight * normalized_test[name]
        for name, weight in zip(best["names"], best["weights"])
    )
    return individual, best, blended_oof, blended_test


def train(args):
    train_data = load_csv(args.train, [args.text_column, "label"])
    test_data = load_csv(args.test, [args.text_column, "id"])
    labels = train_data["label"].astype(int).to_numpy()
    texts = train_data[args.text_column]
    test_texts = test_data[args.text_column]
    groups = texts.str.split().str.join(" ").str.casefold()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(texts, labels, groups=groups))

    names = [spec[0] for spec in SPECS] + ["combined_nb_c05"]
    oof = {name: np.zeros(len(train_data)) for name in names}
    test_scores = {name: np.zeros(len(test_data)) for name in names}
    existing_oof = np.load(args.base / "oof_scores.npy")
    existing_test = np.load(args.base / "test_scores.npy")
    oof["combined_nb_c05"] = existing_oof
    test_scores["combined_nb_c05"] = existing_test

    for fold, (fit_indices, valid_indices) in enumerate(splits, start=1):
        print(f"Fold {fold}/5: loading vectorizers and transforming text...", flush=True)
        bundle = joblib.load(args.base / f"model_fold_{fold}.joblib")
        word = bundle["word_vectorizer"]
        char = bundle["char_vectorizer"]
        word_fit = word.transform(texts.iloc[fit_indices])
        word_valid = word.transform(texts.iloc[valid_indices])
        word_test = word.transform(test_texts)
        char_fit = char.transform(texts.iloc[fit_indices])
        char_valid = char.transform(texts.iloc[valid_indices])
        char_test = char.transform(test_texts)
        matrices = {
            "word": (word_fit, word_valid, word_test),
            "char": (char_fit, char_valid, char_test),
            "combined": (
                hstack([word_fit, char_fit], format="csr"),
                hstack([word_valid, char_valid], format="csr"),
                hstack([word_test, char_test], format="csr"),
            ),
        }
        for name, feature_type, use_nb, c_value in SPECS:
            validation_values, test_values = score_model(
                *matrices[feature_type], labels[fit_indices], use_nb, c_value
            )
            oof[name][valid_indices] = validation_values
            test_scores[name] += test_values / 5
        print(f"Fold {fold}/5 complete", flush=True)

    individual, best, blended_oof, blended_test = search_blends(
        labels, oof, test_scores
    )
    predictions = (blended_test >= best["threshold"]).astype(int)
    submission = pd.DataFrame({"id": test_data["id"], "label": predictions})
    args.output.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output / "submission.csv", index=False)
    np.save(args.output / "oof_scores.npy", blended_oof)
    np.save(args.output / "test_scores.npy", blended_test)
    metrics = {
        "seed": SEED,
        "method": "five_fold_text_only_diverse_svm_blend",
        "uses_id_as_feature": False,
        "individual": individual,
        "selected_blend": best,
        "test_label_counts": {
            str(key): int(value)
            for key, value in submission["label"].value_counts().sort_index().items()
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    for name in sorted(individual, key=lambda item: individual[item]["f1"], reverse=True):
        print(name, json.dumps(individual[name], sort_keys=True), flush=True)
    print("Selected blend: " + json.dumps(best, sort_keys=True), flush=True)
    print("Label counts: " + json.dumps(metrics["test_label_counts"]), flush=True)
    print(f"Results saved in {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--base", type=Path, default=Path("artifacts/optimized_cv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/optimized_blend"))
    args = parser.parse_args()
    try:
        train(args)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Training stopped: {error}\n")


if __name__ == "__main__":
    main()
