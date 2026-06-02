import pytest

from src.evaluation.statistical_tests import (
    build_mcnemar_table,
    run_mcnemar_test,
    run_wilcoxon_test,
)


def test_run_wilcoxon_test_returns_statistic_and_p_value():
    automata_scores = [0.10, 0.20, 0.30, 0.25, 0.15]
    dl_scores = [0.20, 0.25, 0.35, 0.30, 0.20]

    result = run_wilcoxon_test(
        model_a_scores=automata_scores,
        model_b_scores=dl_scores,
    )

    assert result["test"] == "wilcoxon_signed_rank"
    assert "statistic" in result
    assert "p_value" in result
    assert result["n_pairs"] == 5


def test_run_wilcoxon_test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        run_wilcoxon_test(
            model_a_scores=[0.1, 0.2],
            model_b_scores=[0.1],
        )


def test_build_mcnemar_table_counts_prediction_disagreements():
    y_true = [0, 0, 1, 1]
    model_a_pred = [0, 0, 1, 0]
    model_b_pred = [0, 1, 1, 1]

    table = build_mcnemar_table(
        y_true=y_true,
        model_a_pred=model_a_pred,
        model_b_pred=model_b_pred,
    )

    assert table == [
        [2, 1],
        [1, 0],
    ]


def test_run_mcnemar_test_returns_p_value():
    y_true = [0, 0, 1, 1]
    model_a_pred = [0, 0, 1, 0]
    model_b_pred = [0, 1, 1, 1]

    result = run_mcnemar_test(
        y_true=y_true,
        model_a_pred=model_a_pred,
        model_b_pred=model_b_pred,
    )

    assert result["test"] == "mcnemar"
    assert "table" in result
    assert "statistic" in result
    assert "p_value" in result