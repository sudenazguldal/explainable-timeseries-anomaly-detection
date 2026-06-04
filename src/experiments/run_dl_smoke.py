from __future__ import annotations

from pathlib import Path


SMOKE_SUMMARY_PATH = Path("reports/results/deep_learning/dl_smoke_summary.json")


def generate_smoke_summary() -> list[dict[str, object]]:
    """
    Builds a lightweight deep learning pipeline smoke summary.
    """
    return []
