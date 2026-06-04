from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.preprocessing.sequence_builder import build_sequence_windows


SMOKE_SUMMARY_PATH = Path("reports/results/deep_learning/dl_smoke_summary.json")
SMOKE_FEATURE_COLUMNS = ["sensor_a", "sensor_b"]
SMOKE_TARGET_COLUMN = "label"
SMOKE_SEQUENCE_LENGTH = 4


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
    }


def generate_smoke_summary() -> list[dict[str, object]]:
    """
    Builds a lightweight deep learning pipeline smoke summary.
    """
    dataframe = build_synthetic_time_series_fixture()
    return [build_sequence_window_smoke_summary(dataframe)]


def _build_smoke_windows(dataframe: pd.DataFrame, channels_first: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    return build_sequence_windows(
        data=dataframe,
        sequence_length=SMOKE_SEQUENCE_LENGTH,
        feature_columns=SMOKE_FEATURE_COLUMNS,
        target_column=SMOKE_TARGET_COLUMN,
        channels_first=channels_first,
    )
