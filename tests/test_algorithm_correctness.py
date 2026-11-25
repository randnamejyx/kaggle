"""Regression checks for score selection without downloading a language model."""

import ast
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

from train_blend_ensemble import search_blends


def transformer_best_f1():
    # Execute the real NumPy-only function without requiring the GPU libraries.
    source = Path(__file__).resolve().parents[1] / "train_transformer.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "best_f1"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["best_f1"]


def test_transformer_threshold_uses_whole_tied_score_groups():
    labels = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.8, 0.8, 0.8])

    result = transformer_best_f1()(labels, scores)

    assert result["threshold"] == 0.9
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["precision"] == 1
    assert result["recall"] == 0.5


def test_transformer_threshold_matches_brute_force_for_quantized_probabilities():
    best_f1 = transformer_best_f1()
    generator = np.random.default_rng(42)
    for _ in range(80):
        labels = generator.integers(0, 2, size=24)
        scores = generator.integers(0, 6, size=24).astype(np.float16) / 5
        result = best_f1(labels, scores)
        attainable = max(
            f1_score(labels, scores >= threshold, zero_division=0)
            for threshold in np.unique(scores)
        )
        assert result["f1"] == pytest.approx(attainable)
        assert result["f1"] == pytest.approx(
            f1_score(labels, scores >= result["threshold"], zero_division=0)
        )


def test_blending_constant_candidate_keeps_informative_candidate():
    labels = np.array([0, 1, 0, 1])
    oof = {
        "constant": np.zeros(4),
        "informative": np.array([-2.0, 2.0, -1.0, 1.0]),
    }
    test = {"constant": np.zeros(2), "informative": np.array([-1.0, 1.0])}

    _, best, blended_oof, blended_test = search_blends(labels, oof, test)

    assert best["f1"] == 1
    assert np.isfinite(blended_oof).all()
    np.testing.assert_array_equal(blended_test >= best["threshold"], [False, True])


def test_blending_all_constant_candidates_has_finite_predictions():
    _, best, blended_oof, blended_test = search_blends(
        np.array([0, 1, 0, 1]),
        {"constant": np.full(4, 3.0)},
        {"constant": np.full(2, 3.0)},
    )

    assert best["f1"] == pytest.approx(2 / 3)
    assert np.isfinite(blended_oof).all()
    np.testing.assert_array_equal(blended_test >= best["threshold"], [True, True])
