from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from prediction_cache import load_score_cache, score_cache_signature
import train_blend_ensemble
import train_charwb_ensemble
import train_cv_ensemble


@pytest.fixture(scope="module")
def cv_experiment(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cv-experiment")
    train = pd.DataFrame([
        {
            "id": str(index),
            "text": ("kind friendly welcome" if index % 2 else "angry hostile insult")
            + f" sample {index}",
            "label": str(index % 2),
        }
        for index in range(60)
    ])
    test = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "text": ["hostile insult", "kind welcome", "angry hostile", "friendly kind"],
    })
    train_path = directory / "train.csv"
    test_path = directory / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    base = directory / "base"
    train_cv_ensemble.train(SimpleNamespace(
        train=train_path, test=test_path, text_column="text", folds=5, c=0.5, output=base,
    ))
    return directory, train, test, base


@pytest.mark.parametrize("module", [train_blend_ensemble, train_charwb_ensemble])
def test_reordered_test_rows_cannot_silently_reuse_scores(cv_experiment, tmp_path, module):
    directory, _, test, base = cv_experiment
    reordered_path = tmp_path / "test.csv"
    test.iloc[::-1].to_csv(reordered_path, index=False)
    args = SimpleNamespace(
        train=directory / "train.csv", test=reordered_path, text_column="text",
        base=base, previous=base, output=tmp_path / "predictions",
    )

    with pytest.raises(ValueError, match="cache"):
        module.train(args)

    assert not (args.output / "submission.csv").exists()


def current_signature(train, test, folds=5):
    groups = train["text"].str.split().str.join(" ").str.casefold()
    splits = list(StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=42,
    ).split(train["text"], train["label"].astype(int), groups))
    return score_cache_signature(train, test, "text", splits)


@pytest.mark.parametrize("change", [
    "train_order", "train_text", "train_label", "test_id", "test_text", "folds",
])
def test_changed_samples_or_folds_cannot_reuse_cache(cv_experiment, change):
    _, original_train, original_test, base = cv_experiment
    train, test = original_train.copy(), original_test.copy()
    folds = 5
    if change == "train_order":
        train = train.iloc[::-1]
    elif change == "train_text":
        train.loc[0, "text"] = "changed training sample"
    elif change == "train_label":
        train.loc[0, "label"] = "1"
    elif change == "test_id":
        test.loc[0, "id"] = "different-id"
    elif change == "test_text":
        test.loc[0, "text"] = "changed test sample"
    elif change == "folds":
        folds = 4

    with pytest.raises(ValueError, match="cache does not match"):
        load_score_cache(base, current_signature(train, test, folds))


def test_old_cache_without_sample_provenance_is_rejected(cv_experiment, tmp_path):
    _, train, test, _ = cv_experiment
    np.save(tmp_path / "oof_scores.npy", np.zeros(len(train)))
    np.save(tmp_path / "test_scores.npy", np.zeros(len(test)))

    with pytest.raises(ValueError, match="cache has no provenance"):
        load_score_cache(tmp_path, current_signature(train, test))


@pytest.mark.parametrize("module", [train_blend_ensemble, train_charwb_ensemble])
def test_matching_cv_cache_runs_complete_fusion(cv_experiment, tmp_path, module):
    directory, train, test, base = cv_experiment
    args = SimpleNamespace(
        train=directory / "train.csv", test=directory / "test.csv", text_column="text",
        base=base, previous=base, output=tmp_path / "predictions",
    )

    module.train(args)

    submission = pd.read_csv(args.output / "submission.csv", dtype={"id": str})
    assert submission.columns.tolist() == ["id", "label"]
    assert submission["id"].tolist() == test["id"].tolist()
    assert submission["label"].isin([0, 1]).all()
    oof, scores = load_score_cache(args.output, current_signature(train, test))
    assert len(oof) == len(train)
    assert len(scores) == len(test)


def test_mismatched_fold_vectorizer_is_rejected(cv_experiment, tmp_path):
    directory, train, test, base = cv_experiment
    bundle = joblib.load(base / "model_fold_1.joblib")
    bundle["score_cache_signature"] = {"different": "experiment"}
    joblib.dump(bundle, tmp_path / "model_fold_1.joblib")
    # Use real matching scores but a vectorizer from an incompatible experiment.
    from prediction_cache import save_score_cache
    signature = current_signature(train, test)
    oof, scores = load_score_cache(base, signature)
    save_score_cache(tmp_path, oof, scores, signature)

    with pytest.raises(ValueError, match="fold model does not match"):
        train_blend_ensemble.train(SimpleNamespace(
            train=directory / "train.csv", test=directory / "test.csv", text_column="text",
            base=tmp_path, output=tmp_path / "predictions",
        ))
