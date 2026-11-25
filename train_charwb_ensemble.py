"""Train and blend five-fold word + boundary-character text models."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from prediction_cache import load_score_cache, save_score_cache, score_cache_signature
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

from train_blend_ensemble import search_blends
from train_optimized import SEED, load_csv, nb_ratio


SPECS = (
    ("charwb_combined_nb_c025", "combined", True, 0.25),
    ("charwb_combined_nb_c05", "combined", True, 0.5),
    ("charwb_combined_plain_c025", "combined", False, 0.25),
    ("charwb_only_nb_c025", "char", True, 0.25),
    ("charwb_only_plain_c025", "char", False, 0.25),
)


def vectorizers():
    return (
        TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 3),
            min_df=2,
            max_features=200_000,
            sublinear_tf=True,
            strip_accents="unicode",
        ),
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 6),
            min_df=2,
            max_features=200_000,
            sublinear_tf=True,
        ),
    )


def train(args):
    train_data = load_csv(args.train, [args.text_column, "label"])
    test_data = load_csv(args.test, [args.text_column, "id"])
    labels = train_data["label"].astype(int).to_numpy()
    texts = train_data[args.text_column]
    test_texts = test_data[args.text_column]
    groups = texts.str.split().str.join(" ").str.casefold()
    splits = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            texts, labels, groups=groups
        )
    )
    names = [spec[0] for spec in SPECS] + ["previous_combined_nb"]
    oof = {name: np.zeros(len(train_data)) for name in names}
    test_scores = {name: np.zeros(len(test_data)) for name in names}
    signature = score_cache_signature(train_data, test_data, args.text_column, splits)
    oof["previous_combined_nb"], test_scores["previous_combined_nb"] = load_score_cache(
        args.previous, signature
    )

    for fold, (fit_indices, valid_indices) in enumerate(splits, start=1):
        print(f"Fold {fold}/5: fitting word13 + char_wb26 TF-IDF...", flush=True)
        word, char = vectorizers()
        word_fit = word.fit_transform(texts.iloc[fit_indices])
        word_valid = word.transform(texts.iloc[valid_indices])
        word_test = word.transform(test_texts)
        char_fit = char.fit_transform(texts.iloc[fit_indices])
        char_valid = char.transform(texts.iloc[valid_indices])
        char_test = char.transform(test_texts)
        matrices = {
            "char": (char_fit, char_valid, char_test),
            "combined": (
                hstack([word_fit, char_fit], format="csr"),
                hstack([word_valid, char_valid], format="csr"),
                hstack([word_test, char_test], format="csr"),
            ),
        }
        for name, feature_type, use_nb, c_value in SPECS:
            fit_matrix, valid_matrix, test_matrix = matrices[feature_type]
            ratio = nb_ratio(fit_matrix, labels[fit_indices]) if use_nb else None
            if ratio is not None:
                classifier_fit = fit_matrix.multiply(ratio)
                classifier_valid = valid_matrix.multiply(ratio)
                classifier_test = test_matrix.multiply(ratio)
            else:
                classifier_fit = fit_matrix
                classifier_valid = valid_matrix
                classifier_test = test_matrix
            classifier = LinearSVC(C=c_value, max_iter=5000, random_state=SEED)
            classifier.fit(classifier_fit, labels[fit_indices])
            oof[name][valid_indices] = classifier.decision_function(classifier_valid)
            test_scores[name] += classifier.decision_function(classifier_test) / 5
        print(f"Fold {fold}/5 complete", flush=True)

    individual, best, blended_oof, blended_test = search_blends(
        labels, oof, test_scores
    )
    predictions = (blended_test >= best["threshold"]).astype(int)
    submission = pd.DataFrame({"id": test_data["id"], "label": predictions})
    args.output.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output / "submission.csv", index=False)
    save_score_cache(args.output, blended_oof, blended_test, signature)
    metrics = {
        "seed": SEED,
        "method": "five_fold_text_only_word13_charwb26_blend",
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
    parser.add_argument("--previous", type=Path, default=Path("artifacts/optimized_cv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/optimized_charwb"))
    args = parser.parse_args()
    try:
        train(args)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Training stopped: {error}\n")


if __name__ == "__main__":
    main()
