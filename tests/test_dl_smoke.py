import json

from src.experiments.run_dl_smoke import export_smoke_summary, generate_smoke_summary


def test_generate_smoke_summary_contains_expected_fields():
    summary = generate_smoke_summary()

    assert {row["model_name"] for row in summary} == {
        "sequence_windows",
        "lstm",
        "cnn1d",
        "metric_calculation",
    }
    for row in summary:
        assert "model_name" in row
        assert "window_count" in row
        assert "feature_count" in row
        assert "output_shape" in row
        assert "metrics" in row

    lstm_summary = next(row for row in summary if row["model_name"] == "lstm")
    cnn_summary = next(row for row in summary if row["model_name"] == "cnn1d")
    metric_summary = next(row for row in summary if row["model_name"] == "metric_calculation")

    assert lstm_summary["output_shape"] == [2, 1]
    assert cnn_summary["output_shape"] == [2, 1]
    assert set(metric_summary["metrics"]) == {"accuracy", "precision", "recall", "f1_score"}


def test_export_smoke_summary_writes_json_artifact(tmp_path):
    summary = generate_smoke_summary()
    output_path = tmp_path / "dl_smoke_summary.json"

    export_smoke_summary(summary, output_path=output_path)

    with output_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    assert payload == summary
