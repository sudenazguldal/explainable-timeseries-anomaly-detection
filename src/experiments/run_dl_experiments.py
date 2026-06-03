from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

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
    classification_threshold: float
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


@dataclass(frozen=True)
class ModelRunContext:
    """
    Model-specific datasets and factory for one dataset split.
    """

    dataset_name: str
    split_name: str
    model_name: str
    train_dataset: TimeSeriesWindowDataset
    validation_dataset: TimeSeriesWindowDataset
    test_dataset: TimeSeriesWindowDataset
    model_factory: Callable[[int], nn.Module]


@dataclass(frozen=True)
class TrainingRunResult:
    """
    Cross-seed training outputs for one model run context.
    """

    context: ModelRunContext
    seed_results: list


@dataclass(frozen=True)
class EvaluationRecord:
    """
    Test-set predictions and metrics for one trained seed.
    """

    dataset_name: str
    split_name: str
    model_name: str
    seed: int
    best_epoch: int
    best_validation_loss: float
    metrics: dict[str, float]
    confusion_matrix: list[list[int]]
    y_true: list[int]
    y_pred: list[int]
    y_scores: list[float]


@dataclass(frozen=True)
class PlotArtifactRecord:
    """
    Saved plot paths for one evaluation record.
    """

    dataset_name: str
    split_name: str
    model_name: str
    seed: int
    confusion_matrix_path: str
    precision_recall_curve_path: str | None
    roc_curve_path: str | None


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
        classification_threshold=float(deep_learning_section.get("classification_threshold", 0.5)),
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


def build_model_run_contexts(
    run_config: ExperimentRunConfig,
    prepared_datasets: dict[str, PreparedExperimentDataset],
) -> list[ModelRunContext]:
    """
    Creates model factories and selects model-compatible datasets for every split.
    """
    model_runs: list[ModelRunContext] = []

    for dataset_name, prepared_dataset in prepared_datasets.items():
        for prepared_split in prepared_dataset.splits:
            for model_name in run_config.model_names:
                train_dataset, validation_dataset, test_dataset = _select_model_datasets(model_name, prepared_split)
                model_factory = _build_model_factory(
                    run_config=run_config,
                    model_name=model_name,
                    train_dataset=train_dataset,
                )
                model_runs.append(
                    ModelRunContext(
                        dataset_name=dataset_name,
                        split_name=prepared_split.split_name,
                        model_name=model_name,
                        train_dataset=train_dataset,
                        validation_dataset=validation_dataset,
                        test_dataset=test_dataset,
                        model_factory=model_factory,
                    )
                )

    return model_runs


def run_cross_seed_training(
    run_config: ExperimentRunConfig,
    model_runs: list[ModelRunContext],
) -> list[TrainingRunResult]:
    """
    Executes the mandatory cross-seed training loop for all model contexts.
    """
    training_results: list[TrainingRunResult] = []

    for model_run in model_runs:
        LOGGER.info(
            "Running cross-seed training | dataset=%s | split=%s | model=%s",
            model_run.dataset_name,
            model_run.split_name,
            model_run.model_name,
        )
        training_config = _with_checkpoint_output_dir(run_config.training_config, run_config.results_dir, model_run)
        seed_results = train_model_across_seeds(
            model_factory=model_run.model_factory,
            train_dataset=model_run.train_dataset,
            validation_dataset=model_run.validation_dataset,
            config=training_config,
            model_name=model_run.model_name,
        )
        training_results.append(TrainingRunResult(context=model_run, seed_results=seed_results))

    return training_results


def evaluate_trained_models(
    run_config: ExperimentRunConfig,
    training_results: list[TrainingRunResult],
) -> list[EvaluationRecord]:
    """
    Evaluates every best seed checkpoint on the held-out test sequence windows.
    """
    evaluation_records: list[EvaluationRecord] = []

    for training_result in training_results:
        model_run = training_result.context
        for seed_result in training_result.seed_results:
            model = model_run.model_factory(seed_result.seed)
            model.load_state_dict(seed_result.best_model_state)
            y_true, y_scores = _predict_anomaly_scores(
                model=model,
                test_dataset=model_run.test_dataset,
                batch_size=run_config.training_config.batch_size,
            )
            y_pred = (np.asarray(y_scores) >= run_config.classification_threshold).astype(int).tolist()
            metrics = calculate_classification_metrics(y_true, y_pred)
            confusion_matrix_values = calculate_confusion_matrix(y_true, y_pred)
            evaluation_records.append(
                EvaluationRecord(
                    dataset_name=model_run.dataset_name,
                    split_name=model_run.split_name,
                    model_name=model_run.model_name,
                    seed=seed_result.seed,
                    best_epoch=seed_result.best_epoch,
                    best_validation_loss=seed_result.best_validation_loss,
                    metrics=metrics,
                    confusion_matrix=confusion_matrix_values,
                    y_true=y_true,
                    y_pred=y_pred,
                    y_scores=y_scores,
                )
            )

    return evaluation_records


def generate_evaluation_plots(
    run_config: ExperimentRunConfig,
    evaluation_records: list[EvaluationRecord],
) -> list[PlotArtifactRecord]:
    """
    Generates confusion matrix, Precision-Recall, and ROC plots for evaluation records.
    """
    plot_records: list[PlotArtifactRecord] = []

    for record in evaluation_records:
        artifact_stem = _build_artifact_stem(record)
        figure_dir = run_config.figures_dir / record.dataset_name / record.split_name / record.model_name
        confusion_matrix_path = save_confusion_matrix_heatmap(
            y_true=record.y_true,
            y_pred=record.y_pred,
            save_path=figure_dir / f"{artifact_stem}_confusion_matrix.png",
        )
        precision_recall_curve_path = _try_save_precision_recall_curve(record, figure_dir, artifact_stem)
        roc_curve_path = _try_save_roc_curve(record, figure_dir, artifact_stem)
        plot_records.append(
            PlotArtifactRecord(
                dataset_name=record.dataset_name,
                split_name=record.split_name,
                model_name=record.model_name,
                seed=record.seed,
                confusion_matrix_path=str(confusion_matrix_path),
                precision_recall_curve_path=precision_recall_curve_path,
                roc_curve_path=roc_curve_path,
            )
        )

    return plot_records


def _try_save_precision_recall_curve(
    record: EvaluationRecord,
    figure_dir: Path,
    artifact_stem: str,
) -> str | None:
    try:
        output_path = save_precision_recall_curve(
            y_true=record.y_true,
            y_scores=record.y_scores,
            save_path=figure_dir / f"{artifact_stem}_precision_recall_curve.png",
        )
        return str(output_path)
    except ValueError as error:
        LOGGER.warning(
            "Skipping Precision-Recall curve | dataset=%s | split=%s | model=%s | seed=%s | reason=%s",
            record.dataset_name,
            record.split_name,
            record.model_name,
            record.seed,
            error,
        )
        return None


def _try_save_roc_curve(
    record: EvaluationRecord,
    figure_dir: Path,
    artifact_stem: str,
) -> str | None:
    try:
        output_path = save_roc_curve(
            y_true=record.y_true,
            y_scores=record.y_scores,
            save_path=figure_dir / f"{artifact_stem}_roc_curve.png",
        )
        return str(output_path)
    except ValueError as error:
        LOGGER.warning(
            "Skipping ROC curve | dataset=%s | split=%s | model=%s | seed=%s | reason=%s",
            record.dataset_name,
            record.split_name,
            record.model_name,
            record.seed,
            error,
        )
        return None


def _predict_anomaly_scores(
    model: nn.Module,
    test_dataset: TimeSeriesWindowDataset,
    batch_size: int,
) -> tuple[list[int], list[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()
    labels: list[int] = []
    scores: list[float] = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs.to(device))
            probabilities = torch.sigmoid(outputs).detach().cpu().reshape(-1).numpy()
            scores.extend(probabilities.astype(float).tolist())
            labels.extend(targets.detach().cpu().reshape(-1).numpy().astype(int).tolist())

    return labels, scores


def _build_artifact_stem(record: EvaluationRecord) -> str:
    return f"{record.dataset_name}_{record.split_name}_{record.model_name}_seed_{record.seed}"


def _with_checkpoint_output_dir(
    training_config: DeepLearningTrainingConfig,
    results_dir: Path,
    model_run: ModelRunContext,
) -> DeepLearningTrainingConfig:
    checkpoint_dir = (
        results_dir
        / "checkpoints"
        / model_run.dataset_name
        / model_run.split_name
        / model_run.model_name
    )

    return replace(training_config, output_dir=checkpoint_dir)


def _select_model_datasets(
    model_name: str,
    prepared_split: PreparedDatasetSplit,
) -> tuple[TimeSeriesWindowDataset, TimeSeriesWindowDataset, TimeSeriesWindowDataset]:
    normalized_name = model_name.strip().lower()

    if normalized_name == "lstm":
        return (
            prepared_split.lstm_train_dataset,
            prepared_split.lstm_validation_dataset,
            prepared_split.lstm_test_dataset,
        )

    if normalized_name == "cnn1d":
        return (
            prepared_split.cnn_train_dataset,
            prepared_split.cnn_validation_dataset,
            prepared_split.cnn_test_dataset,
        )

    raise ValueError(f"Unsupported deep learning model: {model_name}")


def _build_model_factory(
    run_config: ExperimentRunConfig,
    model_name: str,
    train_dataset: TimeSeriesWindowDataset,
) -> Callable[[int], nn.Module]:
    input_shape = _infer_input_shape(model_name, train_dataset)
    model_config = _get_model_config(run_config.project_config, model_name)

    def factory(seed: int) -> nn.Module:
        LOGGER.info("Creating model | model=%s | seed=%s", model_name, seed)
        return create_deep_learning_model(
            model_name=model_name,
            input_shape=input_shape,
            model_config=model_config,
            output_size=1,
        )

    return factory


def _infer_input_shape(model_name: str, train_dataset: TimeSeriesWindowDataset) -> tuple[int, int]:
    if len(train_dataset.windows) == 0:
        raise ValueError(f"Training dataset for {model_name} cannot be empty.")

    window_shape = train_dataset.windows.shape
    normalized_name = model_name.strip().lower()

    if normalized_name == "lstm":
        return int(window_shape[1]), int(window_shape[2])

    if normalized_name == "cnn1d":
        return int(window_shape[2]), int(window_shape[1])

    raise ValueError(f"Unsupported deep learning model: {model_name}")


def _get_model_config(project_config: Mapping[str, object], model_name: str) -> Mapping[str, object]:
    deep_learning_section = _get_config_section(project_config, "deep_learning")
    model_configs = deep_learning_section.get("model_configs", {})
    if not isinstance(model_configs, Mapping):
        raise ValueError("deep_learning.model_configs must be a mapping when provided.")

    model_config = model_configs.get(model_name, {})
    if not isinstance(model_config, Mapping):
        raise ValueError(f"Model configuration must be a mapping: {model_name}")

    return model_config


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
