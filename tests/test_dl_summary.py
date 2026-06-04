import pytest

from src.experiments.summarize_dl_results import summarize_evaluation_records


def test_summarize_evaluation_records_aggregates_metrics_by_dataset_and_model():
    records = [
        {
            "dataset_name": "skab",
            "model_name": "lstm",
            "metrics": {
                "accuracy": 0.8,
                "precision": 0.5,
                "recall": 1.0,
                "f1_score": 0.6666666667,
            },
        },
        {
            "dataset_name": "skab",
            "model_name": "lstm",
            "metrics": {
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1_score": 1.0,
            },
        },
        {
            "dataset_name": "batadal",
            "model_name": "cnn1d",
            "metrics": {
                "accuracy": 0.25,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            },
        },
    ]

    summary_rows = summarize_evaluation_records(records)

    skab_lstm = next(
        row for row in summary_rows if row["dataset_name"] == "skab" and row["model_name"] == "lstm"
    )
    assert skab_lstm["run_count"] == 2
    assert skab_lstm["accuracy_mean"] == pytest.approx(0.9)
    assert skab_lstm["accuracy_std"] == pytest.approx(0.1414213562)
    assert skab_lstm["precision_mean"] == pytest.approx(0.75)
    assert skab_lstm["recall_std"] == pytest.approx(0.0)
    assert skab_lstm["f1_score_mean"] == pytest.approx(0.83333333335)

    batadal_cnn = next(
        row for row in summary_rows if row["dataset_name"] == "batadal" and row["model_name"] == "cnn1d"
    )
    assert batadal_cnn["run_count"] == 1
    assert batadal_cnn["accuracy_std"] == 0.0
