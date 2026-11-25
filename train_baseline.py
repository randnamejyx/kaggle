"""Train a CPU text classifier and export a Kaggle submission."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, make_pipeline


def load_csv(path, required):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}. Found {list(frame.columns)}.")
    if frame.empty:
        raise ValueError(f"{path}: no rows.")
    return frame


def check_ids(ids, path):
    if ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError(f"{path}: id values must be nonempty and unique.")


def train(args):
    train_data = load_csv(args.train, [args.text_column, "label"])
    test_data = load_csv(args.test, [args.text_column, "id"])
    check_ids(test_data["id"], args.test)

    texts = train_data[args.text_column]
    labels = train_data["label"].str.strip()
    if set(labels) != {"0", "1"}:
        raise ValueError(f"{args.train}: label must contain both classes 0 and 1.")
    labels = labels.astype(int)

    sample = None
    if args.sample is not None:
        sample = load_csv(args.sample, ["id"])
        check_ids(sample["id"], args.sample)
        if len(sample) != len(test_data):
            raise ValueError(
                f"{args.sample}: expected {len(test_data)} rows, found {len(sample)}."
            )

    groups = texts.str.split().str.join(" ").str.casefold()
    folds = min(5, int(groups.groupby(labels).nunique().min()))
    if folds < 2:
        raise ValueError("Validation needs at least two distinct texts per class.")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    for fit, validation in splitter.split(texts, labels, groups=groups):
        if labels.iloc[fit].nunique() == labels.iloc[validation].nunique() == 2:
            break
    else:
        raise ValueError("Could not split the data into two sets that both contain both labels.")

    model = make_pipeline(
        FeatureUnion([
            ("word", TfidfVectorizer(
                token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), max_features=100_000
            )),
            ("char", TfidfVectorizer(
                analyzer="char", ngram_range=(2, 5), max_features=100_000
            )),
        ]),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(texts.iloc[fit], labels.iloc[fit])
    truth = labels.iloc[validation]
    predictions = model.predict(texts.iloc[validation])
    metrics = {
        "seed": 42,
        "train_rows": len(train_data),
        "validation_rows": len(validation),
        "validation_positive_rows": int(truth.sum()),
        "test_rows": len(test_data),
        "f1_positive": f1_score(truth, predictions, pos_label=1, zero_division=0),
        "f1_macro": f1_score(truth, predictions, average="macro", zero_division=0),
        "precision_positive": precision_score(truth, predictions, zero_division=0),
        "recall_positive": recall_score(truth, predictions, zero_division=0),
        "confusion_matrix_0_1": confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
    }
    model.fit(texts, labels)
    test_predictions = model.predict(test_data[args.text_column])
    if sample is None or set(sample["id"]) != set(test_data["id"]):
        if sample is not None:
            print(
                f"Warning: {args.sample} IDs do not match {args.test}; "
                "using test IDs because Kaggle scores predictions by test ID."
            )
        submission = pd.DataFrame({
            "id": test_data["id"],
            "label": test_predictions,
        })
    else:
        predictions_by_id = pd.Series(test_predictions, index=test_data["id"])
        submission = pd.DataFrame({
            "id": sample["id"],
            "label": predictions_by_id.loc[sample["id"]].to_numpy(),
        })

    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output / "model.joblib")
    submission.to_csv(args.output / "submission.csv", index=False)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Results saved in {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--sample", type=Path, help="Optional CSV defining submission id order.")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    try:
        train(args)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Training stopped: {exc}\n")


if __name__ == "__main__":
    main()
