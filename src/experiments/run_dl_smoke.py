from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SMOKE_SUMMARY_PATH = Path("reports/results/deep_learning/dl_smoke_summary.json")
SMOKE_FEATURE_COLUMNS = ["sensor_a", "sensor_b"]
SMOKE_TARGET_COLUMN = "label"


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


def generate_smoke_summary() -> list[dict[str, object]]:
    """
    Builds a lightweight deep learning pipeline smoke summary.
    """
    return []
