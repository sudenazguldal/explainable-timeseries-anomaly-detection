from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SequenceWindowConfig:
    """
    Configuration for building overlapping time-series windows.
    """

    sequence_length: int
    stride: int = 1
    target_horizon: int = 0
    channels_first: bool = False
    feature_dtype: torch.dtype = torch.float32
    target_dtype: torch.dtype = torch.float32


class TimeSeriesWindowDataset(Dataset):
    """
    PyTorch dataset for pre-built time-series sequence windows.
    """

    def __init__(self, windows: torch.Tensor, targets: torch.Tensor | None = None):
        if targets is not None and len(windows) != len(targets):
            raise ValueError("Windows and targets must contain the same number of samples.")

        self.windows = windows
        self.targets = targets

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        if self.targets is None:
            return self.windows[index]

        return self.windows[index], self.targets[index]


def build_sequence_config(config: Mapping[str, object] | None = None, **overrides) -> SequenceWindowConfig:
    """
    Builds a sequence window configuration from a dictionary and keyword overrides.
    """
    values = dict(config or {})
    values.update({key: value for key, value in overrides.items() if value is not None})

    if "sequence_length" not in values:
        raise ValueError("sequence_length is required to build sequence windows.")

    return SequenceWindowConfig(
        sequence_length=int(values["sequence_length"]),
        stride=int(values.get("stride", 1)),
        target_horizon=int(values.get("target_horizon", 0)),
        channels_first=bool(values.get("channels_first", False)),
    )


def build_sequence_windows(
    data: pd.DataFrame | pd.Series | np.ndarray | Sequence[float],
    sequence_length: int,
    feature_columns: Sequence[str] | None = None,
    target_column: str | None = None,
    target_values: pd.Series | np.ndarray | Sequence[float] | None = None,
    stride: int = 1,
    target_horizon: int = 0,
    channels_first: bool = False,
    feature_dtype: torch.dtype = torch.float32,
    target_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Converts sequential data into overlapping windows.

    Returned feature windows use shape (samples, sequence_length, features) for LSTM
    inputs. Set channels_first=True to return (samples, features, sequence_length),
    which is the expected layout for torch.nn.Conv1d.

    If targets are provided, target_horizon=0 aligns each target with the last time
    step in its window. Positive horizons select future targets.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than zero.")
    if stride <= 0:
        raise ValueError("stride must be greater than zero.")
    if target_horizon < 0:
        raise ValueError("target_horizon cannot be negative.")

    feature_array = _to_2d_float_array(data, feature_columns, target_column)
    target_array = _resolve_target_array(data, target_column, target_values)
    sample_count = len(feature_array)
    max_target_index = sample_count - 1 if target_array is None else min(sample_count, len(target_array)) - 1
    last_start = max_target_index - target_horizon - sequence_length + 1

    if last_start < 0:
        empty_windows = torch.empty(
            (0, feature_array.shape[1], sequence_length)
            if channels_first
            else (0, sequence_length, feature_array.shape[1]),
            dtype=feature_dtype,
        )
        empty_targets = None
        if target_array is not None:
            empty_targets = torch.empty((0,), dtype=target_dtype)
        return empty_windows, empty_targets

    window_starts = range(0, last_start + 1, stride)
    windows = np.stack(
        [feature_array[start : start + sequence_length] for start in window_starts],
        axis=0,
    )

    if channels_first:
        windows = np.transpose(windows, (0, 2, 1))

    window_tensor = torch.as_tensor(windows, dtype=feature_dtype)

    if target_array is None:
        return window_tensor, None

    target_indices = [start + sequence_length - 1 + target_horizon for start in window_starts]
    targets = target_array[target_indices]
    target_tensor = torch.as_tensor(targets, dtype=target_dtype)

    return window_tensor, target_tensor


def build_window_dataset(
    data: pd.DataFrame | pd.Series | np.ndarray | Sequence[float],
    config: SequenceWindowConfig,
    feature_columns: Sequence[str] | None = None,
    target_column: str | None = None,
    target_values: pd.Series | np.ndarray | Sequence[float] | None = None,
) -> TimeSeriesWindowDataset:
    """
    Builds a PyTorch dataset from sequence window configuration.
    """
    windows, targets = build_sequence_windows(
        data=data,
        sequence_length=config.sequence_length,
        feature_columns=feature_columns,
        target_column=target_column,
        target_values=target_values,
        stride=config.stride,
        target_horizon=config.target_horizon,
        channels_first=config.channels_first,
        feature_dtype=config.feature_dtype,
        target_dtype=config.target_dtype,
    )

    return TimeSeriesWindowDataset(windows, targets)


def _to_2d_float_array(
    data: pd.DataFrame | pd.Series | np.ndarray | Sequence[float],
    feature_columns: Sequence[str] | None,
    target_column: str | None,
) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        columns = list(feature_columns) if feature_columns is not None else _infer_feature_columns(data, target_column)
        missing_columns = [column for column in columns if column not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing feature columns: {missing_columns}")

        values = data[columns].to_numpy()
    elif isinstance(data, pd.Series):
        values = data.to_numpy()
    else:
        values = np.asarray(data)

    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError("Input data must be one-dimensional or two-dimensional.")
    if len(values) == 0:
        raise ValueError("Input data cannot be empty.")

    return values.astype(np.float32, copy=False)


def _infer_feature_columns(data: pd.DataFrame, target_column: str | None) -> list[str]:
    excluded_columns = {target_column} if target_column is not None else set()
    columns = [column for column in data.columns if column not in excluded_columns]

    if not columns:
        raise ValueError("No feature columns are available for sequence windows.")

    return columns


def _resolve_target_array(
    data: pd.DataFrame | pd.Series | np.ndarray | Sequence[float],
    target_column: str | None,
    target_values: pd.Series | np.ndarray | Sequence[float] | None,
) -> np.ndarray | None:
    if target_values is not None:
        values = target_values.to_numpy() if isinstance(target_values, pd.Series) else np.asarray(target_values)
    elif target_column is not None:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("target_column can only be used when data is a DataFrame.")
        if target_column not in data.columns:
            raise ValueError(f"Missing target column: {target_column}")
        values = data[target_column].to_numpy()
    else:
        return None

    if values.ndim > 2:
        raise ValueError("Targets must be one-dimensional or two-dimensional.")
    if len(values) == 0:
        raise ValueError("Targets cannot be empty.")

    return values
