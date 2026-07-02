import torch

from src.models.train_deep_learning import compute_pos_weight
from src.preprocessing.sequence_builder import TimeSeriesWindowDataset


def _build_dataset(targets: list[float]) -> TimeSeriesWindowDataset:
    windows = torch.zeros((len(targets), 2, 1))
    return TimeSeriesWindowDataset(windows, torch.tensor(targets, dtype=torch.float32))


def test_compute_pos_weight_returns_negative_over_positive_ratio():
    # 9 normal samples, 1 anomaly -> pos_weight should upweight the anomaly class by 9x.
    dataset = _build_dataset([0.0] * 9 + [1.0])

    pos_weight = compute_pos_weight(dataset)

    assert pos_weight is not None
    assert pos_weight.item() == 9.0


def test_compute_pos_weight_returns_none_when_only_one_class_present():
    dataset = _build_dataset([0.0] * 5)

    pos_weight = compute_pos_weight(dataset)

    assert pos_weight is None


def test_compute_pos_weight_returns_none_when_dataset_has_no_targets():
    windows = torch.zeros((5, 2, 1))
    dataset = TimeSeriesWindowDataset(windows, None)

    pos_weight = compute_pos_weight(dataset)

    assert pos_weight is None


def test_compute_pos_weight_is_one_for_balanced_classes():
    dataset = _build_dataset([0.0, 0.0, 1.0, 1.0])

    pos_weight = compute_pos_weight(dataset)

    assert pos_weight is not None
    assert pos_weight.item() == 1.0
