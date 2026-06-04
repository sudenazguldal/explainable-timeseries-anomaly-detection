from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import calculate_classification_metrics
from src.models.cnn1d_model import CNN1DModel
from src.models.lstm_model import LSTMModel
from src.preprocessing.sequence_builder import build_sequence_windows


SMOKE_SUMMARY_PATH = Path("reports/results/deep_learning/dl_smoke_summary.json")
SMOKE_FEATURE_COLUMNS = ["sensor_a", "sensor_b"]
SMOKE_TARGET_COLUMN = "label"
SMOKE_SEQUENCE_LENGTH = 4
SMOKE_CLASSIFICATION_THRESHOLD = 0.5


def build_synthetic_time_series_fixture(row_count: int = 12) -> pd.DataFrame:
    """
    Builds a deterministic toy time-series frame for fast DL smoke checks.
    """
    if row_count <= 0:
        raise ValueError("row_count must be greater than zero.")

    time_index = np.arange(row_count, dtype=np.float32)
    return pd.DataFrame(
        {
            "sensor_a": time_index / max(row_count - 1, 1),
            "sensor_b": np.sin(time_index / 2.0),
            "label": (time_index >= row_count // 2).astype(int),
        }
    )


def build_sequence_window_smoke_summary(dataframe: pd.DataFrame) -> dict[str, object]:
    """
    Validates LSTM-style sequence window creation on the synthetic fixture.
    """
    windows, targets = _build_smoke_windows(dataframe, channels_first=False)
    if targets is None:
        raise ValueError("Smoke fixture must produce target windows.")

    return {
        "model_name": "sequence_windows",
        "window_count": int(windows.shape[0]),
        "feature_count": int(windows.shape[2]),
        "target_count": int(targets.shape[0]),
        "output_shape": list(windows.shape),
        "metrics": {},
    }


def build_lstm_forward_smoke_summary(dataframe: pd.DataFrame) -> dict[str, object]:
    """
    Validates that the LSTM baseline can score a small sequence batch.
    """
    windows, _ = _build_smoke_windows(dataframe, channels_first=False)
    model = LSTMModel(input_size=len(SMOKE_FEATURE_COLUMNS), hidden_size=4, output_size=1)
    model.eval()

    with torch.no_grad():
        outputs = model(windows[:2])

    return {
        "model_name": "lstm",
        "window_count": int(windows.shape[0]),
        "feature_count": int(windows.shape[2]),
        "output_shape": list(outputs.shape),
        "metrics": {},
    }


def build_cnn1d_forward_smoke_summary(dataframe: pd.DataFrame) -> dict[str, object]:
    """
    Validates that the CNN1D baseline can score channels-first sequence windows.
    """
    windows, _ = _build_smoke_windows(dataframe, channels_first=True)
    model = CNN1DModel(
        input_channels=len(SMOKE_FEATURE_COLUMNS),
        hidden_channels=(4,),
        kernel_sizes=3,
        output_size=1,
    )
    model.eval()

    with torch.no_grad():
        outputs = model(windows[:2])

    return {
        "model_name": "cnn1d",
        "window_count": int(windows.shape[0]),
        "feature_count": int(windows.shape[1]),
        "output_shape": list(outputs.shape),
        "metrics": {},
    }


def build_metric_smoke_summary() -> dict[str, object]:
    """
    Validates binary metric calculation on deterministic fake prediction scores.
    """
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.7, 0.8, 0.3]
    y_pred = [int(score >= SMOKE_CLASSIFICATION_THRESHOLD) for score in y_scores]

    return {
        "model_name": "metric_calculation",
        "window_count": 0,
        "feature_count": 0,
        "output_shape": [],
        "metrics": calculate_classification_metrics(y_true, y_pred),
    }


def generate_smoke_summary() -> list[dict[str, object]]:
    """
    Builds a lightweight deep learning pipeline smoke summary.
    """
    dataframe = build_synthetic_time_series_fixture()
    return [
        build_sequence_window_smoke_summary(dataframe),
        build_lstm_forward_smoke_summary(dataframe),
        build_cnn1d_forward_smoke_summary(dataframe),
        build_metric_smoke_summary(),
    ]


def export_smoke_summary(
    summary: list[dict[str, object]],
    output_path: Path = SMOKE_SUMMARY_PATH,
) -> Path:
    """
    Writes the smoke summary artifact as JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    return output_path


def _build_smoke_windows(dataframe: pd.DataFrame, channels_first: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    return build_sequence_windows(
        data=dataframe,
        sequence_length=SMOKE_SEQUENCE_LENGTH,
        feature_columns=SMOKE_FEATURE_COLUMNS,
        target_column=SMOKE_TARGET_COLUMN,
        channels_first=channels_first,
    )
