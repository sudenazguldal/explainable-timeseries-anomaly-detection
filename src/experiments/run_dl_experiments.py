from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config.load_config import load_config
from src.data.batadal_loader import (
    detect_batadal_target_column,
    detect_batadal_time_column,
    get_batadal_feature_columns,
    load_batadal_dataset,
    split_batadal_time_ordered,
)
from src.data.skab_loader import get_skab_feature_columns, load_skab_dataset
from src.data.splitter import create_skab_group_folds
from src.evaluation.metrics import calculate_classification_metrics, calculate_confusion_matrix
from src.evaluation.plots import save_confusion_matrix_heatmap, save_precision_recall_curve, save_roc_curve
from src.models.train_deep_learning import (
    DeepLearningTrainingConfig,
    build_training_config,
    create_deep_learning_model,
    train_model_across_seeds,
)
from src.preprocessing.scaler import fit_transform_train, transform_with_fitted_scaler
from src.preprocessing.sequence_builder import TimeSeriesWindowDataset, build_sequence_windows


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetRunConfig:
    """
    Parsed dataset-specific settings for deep learning experiments.
    """

    name: str
    target_column: str
    feature_columns: list[str]
    group_column: str | None = None
    time_column: str | None = None


@dataclass(frozen=True)
class ExperimentRunConfig:
    """
    Parsed top-level settings for deep learning experiment execution.
    """

    project_config: Mapping[str, object]
    training_config: DeepLearningTrainingConfig
    sequence_length: int
    normalization_method: str
    model_names: list[str]
    results_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class PreparedDatasetSplit:
    """
    Windowed train, validation, and test datasets for one split.
    """

    split_name: str
    lstm_train_dataset: TimeSeriesWindowDataset
    lstm_validation_dataset: TimeSeriesWindowDataset
    lstm_test_dataset: TimeSeriesWindowDataset
    cnn_train_dataset: TimeSeriesWindowDataset
    cnn_validation_dataset: TimeSeriesWindowDataset
    cnn_test_dataset: TimeSeriesWindowDataset


@dataclass(frozen=True)
class PreparedExperimentDataset:
    """
    Windowed datasets and metadata for one source dataset.
    """

    dataset_config: DatasetRunConfig
    splits: list[PreparedDatasetSplit]


def parse_experiment_config(config_path: str | Path = "config.yaml") -> ExperimentRunConfig:
    """
    Loads and validates configuration values needed by the deep learning runner.
    """
    project_config = load_config(str(config_path))
    deep_learning_section = _get_config_section(project_config, "deep_learning")
    preprocessing_section = _get_config_section(project_config, "preprocessing")
    logging_section = _get_config_section(project_config, "logging")
    training_config = build_training_config(project_config)

    return ExperimentRunConfig(
        project_config=project_config,
        training_config=training_config,
        sequence_length=int(deep_learning_section["sequence_length"]),
        normalization_method=str(preprocessing_section["normalization"]),
        model_names=[str(model_name) for model_name in deep_learning_section["models"]],
        results_dir=Path(str(logging_section["results_dir"])) / "deep_learning",
        figures_dir=Path(str(logging_section["figures_dir"])) / "deep_learning",
    )


def load_experiment_datasets(
    run_config: ExperimentRunConfig,
) -> dict[str, tuple[DatasetRunConfig, list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]]]:
    """
    Loads configured datasets and creates train, validation, and test splits.
    """
    project_config = run_config.project_config

    return {
        "skab": _load_skab_splits(project_config),
        "batadal": _load_batadal_splits(project_config),
    }


def prepare_sequence_datasets(
    run_config: ExperimentRunConfig,
    loaded_datasets: dict[str, tuple[DatasetRunConfig, list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]]],
) -> dict[str, PreparedExperimentDataset]:
    """
    Applies train-only normalization and converts split dataframes into sequence datasets.
    """
    prepared_datasets: dict[str, PreparedExperimentDataset] = {}

    for dataset_name, (dataset_config, split_frames) in loaded_datasets.items():
        prepared_splits = [
            _prepare_sequence_split(
                run_config=run_config,
                dataset_config=dataset_config,
                split_name=f"split_{split_index}",
                train_df=train_df,
                validation_df=validation_df,
                test_df=test_df,
            )
            for split_index, (train_df, validation_df, test_df) in enumerate(split_frames, start=1)
        ]
        prepared_datasets[dataset_name] = PreparedExperimentDataset(
            dataset_config=dataset_config,
            splits=prepared_splits,
        )

    return prepared_datasets


def _prepare_sequence_split(
    run_config: ExperimentRunConfig,
    dataset_config: DatasetRunConfig,
    split_name: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> PreparedDatasetSplit:
    scaled_train_df, scaler = fit_transform_train(
        train_df=train_df,
        feature_columns=dataset_config.feature_columns,
        method=run_config.normalization_method,
    )
    scaled_validation_df = transform_with_fitted_scaler(
        df=validation_df,
        feature_columns=dataset_config.feature_columns,
        scaler=scaler,
    )
    scaled_test_df = transform_with_fitted_scaler(
        df=test_df,
        feature_columns=dataset_config.feature_columns,
        scaler=scaler,
    )

    return PreparedDatasetSplit(
        split_name=split_name,
        lstm_train_dataset=_build_window_dataset_for_model(run_config, dataset_config, scaled_train_df, channels_first=False),
        lstm_validation_dataset=_build_window_dataset_for_model(run_config, dataset_config, scaled_validation_df, channels_first=False),
        lstm_test_dataset=_build_window_dataset_for_model(run_config, dataset_config, scaled_test_df, channels_first=False),
        cnn_train_dataset=_build_window_dataset_for_model(run_config, dataset_config, scaled_train_df, channels_first=True),
        cnn_validation_dataset=_build_window_dataset_for_model(run_config, dataset_config, scaled_validation_df, channels_first=True),
        cnn_test_dataset=_build_window_dataset_for_model(run_config, dataset_config, scaled_test_df, channels_first=True),
    )


def _build_window_dataset_for_model(
    run_config: ExperimentRunConfig,
    dataset_config: DatasetRunConfig,
    dataframe: pd.DataFrame,
    channels_first: bool,
) -> TimeSeriesWindowDataset:
    windows, targets = build_sequence_windows(
        data=dataframe,
        sequence_length=run_config.sequence_length,
        feature_columns=dataset_config.feature_columns,
        target_column=dataset_config.target_column,
        channels_first=channels_first,
    )
    if targets is None:
        raise ValueError("Deep learning experiments require target labels.")

    return TimeSeriesWindowDataset(windows, targets)


def _load_skab_splits(project_config: Mapping[str, object]) -> tuple[DatasetRunConfig, list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]]:
    datasets_section = _get_config_section(project_config, "datasets")
    skab_section = _get_nested_config_section(datasets_section, "skab")
    random_seeds = _get_random_seeds(project_config)
    target_column = str(skab_section["target_column"])
    group_column = str(skab_section["group_column"])
    dataframe = load_skab_dataset(
        raw_path=str(skab_section["raw_path"]),
        use_groups=[str(group) for group in skab_section["use_groups"]],
    )
    feature_columns = get_skab_feature_columns(
        df=dataframe,
        target_column=target_column,
        excluded_columns=[str(column) for column in skab_section["excluded_columns"]],
    )
    folds = create_skab_group_folds(
        df=dataframe,
        target_column=target_column,
        group_column=group_column,
        n_splits=len(random_seeds),
        stratified=True,
        random_seed=random_seeds[0],
    )
    validation_ratio = _get_default_validation_ratio(project_config)
    split_frames = [
        _split_skab_fold(dataframe, train_indices, test_indices, validation_ratio=validation_ratio)
        for train_indices, test_indices in folds
    ]

    return (
        DatasetRunConfig(
            name="skab",
            target_column=target_column,
            feature_columns=feature_columns,
            group_column=group_column,
        ),
        split_frames,
    )


def _load_batadal_splits(project_config: Mapping[str, object]) -> tuple[DatasetRunConfig, list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]]:
    datasets_section = _get_config_section(project_config, "datasets")
    batadal_section = _get_nested_config_section(datasets_section, "batadal")
    dataframe = load_batadal_dataset(raw_path=str(batadal_section["raw_path"]))
    target_column = detect_batadal_target_column(
        df=dataframe,
        target_column_candidates=[str(column) for column in batadal_section["target_column_candidates"]],
    )
    time_column = detect_batadal_time_column(
        df=dataframe,
        time_column_candidates=[str(column) for column in batadal_section["time_column_candidates"]],
    )
    feature_columns = get_batadal_feature_columns(
        df=dataframe,
        target_column=target_column,
        time_column=time_column,
    )
    split_section = _get_nested_config_section(batadal_section, "split")
    split_frames = [
        split_batadal_time_ordered(
            df=dataframe,
            train_ratio=float(split_section["train"]),
            validation_ratio=float(split_section["validation"]),
            test_ratio=float(split_section["test"]),
            time_column=time_column,
        )
    ]

    return (
        DatasetRunConfig(
            name="batadal",
            target_column=target_column,
            feature_columns=feature_columns,
            time_column=time_column,
        ),
        split_frames,
    )


def _split_skab_fold(
    dataframe: pd.DataFrame,
    train_indices: list[int],
    test_indices: list[int],
    validation_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_validation_df = dataframe.iloc[train_indices].copy().reset_index(drop=True)
    test_df = dataframe.iloc[test_indices].copy().reset_index(drop=True)
    validation_start = int(len(train_validation_df) * (1.0 - validation_ratio))
    train_df = train_validation_df.iloc[:validation_start].copy().reset_index(drop=True)
    validation_df = train_validation_df.iloc[validation_start:].copy().reset_index(drop=True)

    return train_df, validation_df, test_df


def _get_config_section(config: Mapping[str, object], section_name: str) -> Mapping[str, object]:
    section = config.get(section_name)
    if not isinstance(section, Mapping):
        raise ValueError(f"Missing or invalid configuration section: {section_name}")

    return section


def _get_nested_config_section(config: Mapping[str, object], section_name: str) -> Mapping[str, object]:
    section = config.get(section_name)
    if not isinstance(section, Mapping):
        raise ValueError(f"Missing or invalid nested configuration section: {section_name}")

    return section


def _get_random_seeds(project_config: Mapping[str, object]) -> list[int]:
    project_section = _get_config_section(project_config, "project")

    return [int(seed) for seed in project_section["random_seeds"]]


def _get_default_validation_ratio(project_config: Mapping[str, object]) -> float:
    datasets_section = _get_config_section(project_config, "datasets")
    batadal_section = _get_nested_config_section(datasets_section, "batadal")
    split_section = _get_nested_config_section(batadal_section, "split")

    return float(split_section["validation"])
