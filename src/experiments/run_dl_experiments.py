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


def _get_config_section(config: Mapping[str, object], section_name: str) -> Mapping[str, object]:
    section = config.get(section_name)
    if not isinstance(section, Mapping):
        raise ValueError(f"Missing or invalid configuration section: {section_name}")

    return section
