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
