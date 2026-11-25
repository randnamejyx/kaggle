"""Train an optimized text-only model without using the ID leakage rule."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC


SEED = 42


def load_csv(path, required):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path}: no rows")
    return frame


def make_vectorizers():
    return (
        TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            min_df=2,
            max_features=150_000,
            sublinear_tf=True,
            strip_accents="unicode",
        ),
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_features=150_000,
            sublinear_tf=True,
        ),
    )


def fit_features(texts):
    word, char = make_vectorizers()
    matrix = hstack(
        [word.fit_transform(texts), char.fit_transform(texts)],
        format="csr",
    )
    return word, char, matrix


def transform_features(word, char, texts):
    return hstack([word.transform(texts), char.transform(texts)], format="csr")


def nb_ratio(matrix, labels):
    positive = np.asarray(matrix[labels == 1].sum(axis=0)).ravel()
    negative = np.asarray(matrix[labels == 0].sum(axis=0)).ravel()
    positive = (positive + 1.0) / (int((labels == 1).sum()) + 1.0)
    negative = (negative + 1.0) / (int((labels == 0).sum()) + 1.0)
    return np.log(positive / negative)


def best_threshold(truth, scores):
    precision, recall, thresholds = precision_recall_curve(truth, scores)
    denominators = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominators,
        out=np.zeros_like(denominators),
        where=denominators != 0,
    )
    index = int(np.argmax(f1))
    return float(thresholds[index]), float(f1[index])


def evaluate_candidate(name, c_value, train_matrix, train_labels, valid_matrix, truth):
    ratio = None
    if name == "nb_svm":
        ratio = nb_ratio(train_matrix, train_labels)
        fit_matrix = train_matrix.multiply(ratio)
        score_matrix = valid_matrix.multiply(ratio)
    else:
        fit_matrix = train_matrix
        score_matrix = valid_matrix

    classifier = LinearSVC(C=c_value, max_iter=5000, random_state=SEED)
    classifier.fit(fit_matrix, train_labels)
    scores = classifier.decision_function(score_matrix)
    threshold, tuned_f1 = best_threshold(truth, scores)
    default_predictions = (scores >= 0).astype(int)
    tuned_predictions = (scores >= threshold).astype(int)
    result = {
        "model": name,
        "C": c_value,
        "threshold": threshold,
        "f1_default": f1_score(truth, default_predictions, zero_division=0),
        "f1_tuned": tuned_f1,
        "precision_tuned": precision_score(truth, tuned_predictions, zero_division=0),
        "recall_tuned": recall_score(truth, tuned_predictions, zero_division=0),
    }
    return result


def train(args):
    train_data = load_csv(args.train, [args.text_column, "label"])
    test_data = load_csv(args.test, [args.text_column, "id"])
    labels = train_data["label"].str.strip()
    if set(labels) != {"0", "1"}:
        raise ValueError("training labels must contain only both 0 and 1")
    labels = labels.astype(int).to_numpy()
    if test_data["id"].str.strip().eq("").any() or test_data["id"].duplicated().any():
        raise ValueError("test IDs must be nonempty and unique")

    texts = train_data[args.text_column]
    groups = texts.str.split().str.join(" ").str.casefold()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fit_indices, valid_indices = next(splitter.split(texts, labels, groups=groups))
    fit_labels = labels[fit_indices]
    truth = labels[valid_indices]

    print("Fitting validation TF-IDF features...", flush=True)
    word, char, fit_matrix = fit_features(texts.iloc[fit_indices])
    valid_matrix = transform_features(word, char, texts.iloc[valid_indices])
    print(
        f"Feature matrix: train={fit_matrix.shape}, validation={valid_matrix.shape}",
        flush=True,
    )

    candidates = []
    for name in ("svm", "nb_svm"):
        for c_value in (0.25, 0.5, 1.0, 1.5, 2.0):
            result = evaluate_candidate(
                name, c_value, fit_matrix, fit_labels, valid_matrix, truth
            )
            candidates.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    best = max(candidates, key=lambda item: item["f1_tuned"])
    print("Selected: " + json.dumps(best, sort_keys=True), flush=True)

    print("Refitting TF-IDF features on all training rows...", flush=True)
    word, char, full_matrix = fit_features(texts)
    test_matrix = transform_features(word, char, test_data[args.text_column])
    ratio = None
    if best["model"] == "nb_svm":
        ratio = nb_ratio(full_matrix, labels)
        classifier_fit = full_matrix.multiply(ratio)
        classifier_test = test_matrix.multiply(ratio)
    else:
        classifier_fit = full_matrix
        classifier_test = test_matrix
    classifier = LinearSVC(C=best["C"], max_iter=5000, random_state=SEED)
    classifier.fit(classifier_fit, labels)
    test_scores = classifier.decision_function(classifier_test)
    test_predictions = (test_scores >= best["threshold"]).astype(int)

    submission = pd.DataFrame({"id": test_data["id"], "label": test_predictions})
    metrics = {
        "seed": SEED,
        "method": "text_only_tfidf_linear_svm",
        "uses_id_as_feature": False,
        "train_rows": len(train_data),
        "validation_rows": len(valid_indices),
        "test_rows": len(test_data),
        "word_features": len(word.vocabulary_),
        "char_features": len(char.vocabulary_),
        "selected": best,
        "candidates": candidates,
        "test_label_counts": {
            str(key): int(value)
            for key, value in submission["label"].value_counts().sort_index().items()
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output / "submission.csv", index=False)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    joblib.dump(
        {
            "word_vectorizer": word,
            "char_vectorizer": char,
            "classifier": classifier,
            "nb_ratio": ratio,
            "threshold": best["threshold"],
            "text_column": args.text_column,
        },
        args.output / "model.joblib",
    )
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Results saved in {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output", type=Path, default=Path("artifacts/optimized_normal"))
    args = parser.parse_args()
    try:
        train(args)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Training stopped: {error}\n")


if __name__ == "__main__":
    main()
