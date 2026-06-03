from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.cnn1d_model import CNN1DModel
from src.models.lstm_model import LSTMModel


REQUIRED_SEEDS = (42, 123, 2026, 7, 999)
REQUIRED_BATCH_SIZE = 32
REQUIRED_MAX_EPOCHS = 50
REQUIRED_EARLY_STOPPING_PATIENCE = 5

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeepLearningTrainingConfig:
    """
    Configuration for reproducible deep learning baseline training.
    """

    seeds: tuple[int, ...] = REQUIRED_SEEDS
    batch_size: int = REQUIRED_BATCH_SIZE
    max_epochs: int = REQUIRED_MAX_EPOCHS
    early_stopping_patience: int = REQUIRED_EARLY_STOPPING_PATIENCE
    learning_rate: float = 0.001
    min_delta: float = 0.0
    num_workers: int = 0
    pin_memory: bool = False
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if tuple(self.seeds) != REQUIRED_SEEDS:
            raise ValueError(f"Training must use the exact seeds: {list(REQUIRED_SEEDS)}")
        if self.batch_size != REQUIRED_BATCH_SIZE:
            raise ValueError(f"Batch size must be exactly {REQUIRED_BATCH_SIZE}.")
        if self.max_epochs != REQUIRED_MAX_EPOCHS:
            raise ValueError(f"Maximum epochs must be exactly {REQUIRED_MAX_EPOCHS}.")
        if self.early_stopping_patience != REQUIRED_EARLY_STOPPING_PATIENCE:
            raise ValueError(f"Early stopping patience must be exactly {REQUIRED_EARLY_STOPPING_PATIENCE}.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than zero.")
        if self.min_delta < 0.0:
            raise ValueError("min_delta cannot be negative.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")


@dataclass
class EpochMetrics:
    """
    Loss values collected after one training epoch.
    """

    epoch: int
    train_loss: float
    validation_loss: float


@dataclass
class SeedTrainingResult:
    """
    Best model state and loss history for one seed execution.
    """

    seed: int
    best_epoch: int
    best_validation_loss: float
    history: list[EpochMetrics]
    best_model_state: dict[str, torch.Tensor]
    checkpoint_path: Path | None = None


@dataclass
class EarlyStopping:
    """
    Tracks validation-loss improvements and stores the best model state.
    """

    patience: int
    min_delta: float = 0.0
    best_loss: float = float("inf")
    best_epoch: int = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement: int = 0

    def update(self, validation_loss: float, model: nn.Module, epoch: int) -> bool:
        improved = validation_loss < self.best_loss - self.min_delta

        if improved:
            self.best_loss = validation_loss
            self.best_epoch = epoch
            self.best_state = _copy_model_state_to_cpu(model)
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience


def build_training_config(project_config: Mapping[str, object]) -> DeepLearningTrainingConfig:
    """
    Builds and validates training configuration from the project configuration dictionary.
    """
    project_section = _get_mapping(project_config, "project")
    deep_learning_section = _get_mapping(project_config, "deep_learning")
    output_dir_value = deep_learning_section.get("model_output_dir")

    return DeepLearningTrainingConfig(
        seeds=tuple(int(seed) for seed in project_section.get("random_seeds", REQUIRED_SEEDS)),
        batch_size=int(deep_learning_section.get("batch_size", REQUIRED_BATCH_SIZE)),
        max_epochs=int(deep_learning_section.get("max_epochs", REQUIRED_MAX_EPOCHS)),
        early_stopping_patience=int(
            deep_learning_section.get("early_stopping_patience", REQUIRED_EARLY_STOPPING_PATIENCE)
        ),
        learning_rate=float(deep_learning_section.get("learning_rate", 0.001)),
        min_delta=float(deep_learning_section.get("early_stopping_min_delta", 0.0)),
        num_workers=int(deep_learning_section.get("num_workers", 0)),
        pin_memory=bool(deep_learning_section.get("pin_memory", False)),
        output_dir=Path(str(output_dir_value)) if output_dir_value else None,
    )


def create_deep_learning_model(
    model_name: str,
    input_shape: tuple[int, int],
    model_config: Mapping[str, object] | None = None,
    output_size: int = 1,
) -> nn.Module:
    """
    Creates a supported deep learning baseline model from input shape metadata.
    """
    model_values = dict(model_config or {})
    normalized_name = model_name.strip().lower()
    sequence_length, feature_count = input_shape

    if sequence_length <= 0 or feature_count <= 0:
        raise ValueError("input_shape must contain positive sequence and feature dimensions.")

    if normalized_name == "lstm":
        model_values["input_size"] = feature_count
        model_values.setdefault("output_size", output_size)
        return LSTMModel.from_config(model_values)

    if normalized_name == "cnn1d":
        model_values["input_channels"] = feature_count
        model_values.setdefault("output_size", output_size)
        return CNN1DModel.from_config(model_values)

    raise ValueError(f"Unsupported deep learning model: {model_name}")


def train_model_across_seeds(
    model_factory: Callable[[int], nn.Module],
    train_dataset: Dataset,
    validation_dataset: Dataset,
    config: DeepLearningTrainingConfig,
    model_name: str,
    device: torch.device | str | None = None,
    criterion: nn.Module | None = None,
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer] | None = None,
) -> list[SeedTrainingResult]:
    """
    Trains the provided model factory sequentially across all mandatory seeds.
    """
    results: list[SeedTrainingResult] = []

    for seed in config.seeds:
        set_reproducible_seed(seed)
        model = model_factory(seed)
        checkpoint_path = _build_checkpoint_path(config.output_dir, model_name, seed)
        LOGGER.info("Starting deep learning training | model=%s | seed=%s", model_name, seed)

        result = train_single_seed(
            model=model,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            config=config,
            seed=seed,
            device=device,
            criterion=criterion,
            optimizer_factory=optimizer_factory,
            checkpoint_path=checkpoint_path,
        )
        results.append(result)
        LOGGER.info(
            "Finished deep learning training | model=%s | seed=%s | best_epoch=%s | best_validation_loss=%.6f",
            model_name,
            seed,
            result.best_epoch,
            result.best_validation_loss,
        )

    return results


def train_single_seed(
    model: nn.Module,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    config: DeepLearningTrainingConfig,
    seed: int,
    device: torch.device | str | None = None,
    criterion: nn.Module | None = None,
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer] | None = None,
    checkpoint_path: Path | None = None,
) -> SeedTrainingResult:
    """
    Trains one model instance with deterministic loaders and validation-loss early stopping.
    """
    set_reproducible_seed(seed)
    selected_device = torch.device(device) if device is not None else _get_default_device()
    model = model.to(selected_device)
    loss_function = criterion if criterion is not None else nn.BCEWithLogitsLoss()
    optimizer = optimizer_factory(model) if optimizer_factory else torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = _build_data_loader(train_dataset, config, seed=seed, shuffle=True)
    validation_loader = _build_data_loader(validation_dataset, config, seed=seed, shuffle=False)
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, min_delta=config.min_delta)
    history: list[EpochMetrics] = []

    for epoch in range(1, config.max_epochs + 1):
        train_loss = _run_training_epoch(model, train_loader, loss_function, optimizer, selected_device)
        validation_loss = evaluate_loss(model, validation_loader, loss_function, selected_device)
        history.append(EpochMetrics(epoch=epoch, train_loss=train_loss, validation_loss=validation_loss))

        LOGGER.info(
            "Deep learning epoch | seed=%s | epoch=%s | train_loss=%.6f | validation_loss=%.6f",
            seed,
            epoch,
            train_loss,
            validation_loss,
        )

        should_stop = early_stopping.update(validation_loss, model, epoch)
        if should_stop:
            LOGGER.info("Early stopping triggered | seed=%s | epoch=%s", seed, epoch)
            break

    if early_stopping.best_state is None:
        raise RuntimeError("Training finished without storing a best model state.")

    model.load_state_dict(early_stopping.best_state)
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(early_stopping.best_state, checkpoint_path)

    return SeedTrainingResult(
        seed=seed,
        best_epoch=early_stopping.best_epoch,
        best_validation_loss=early_stopping.best_loss,
        history=history,
        best_model_state=early_stopping.best_state,
        checkpoint_path=checkpoint_path,
    )


def evaluate_loss(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Computes average loss without updating model weights.
    """
    model.eval()
    total_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            prepared_targets = _prepare_targets(targets, outputs)
            loss = criterion(outputs, prepared_targets)
            batch_size = inputs.size(0)
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Validation data loader cannot be empty.")

    return total_loss / sample_count


def set_reproducible_seed(seed: int) -> None:
    """
    Sets random seeds and deterministic PyTorch settings for reproducible training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run_training_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        prepared_targets = _prepare_targets(targets, outputs)
        loss = criterion(outputs, prepared_targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Training data loader cannot be empty.")

    return total_loss / sample_count


def _build_data_loader(dataset: Dataset, config: DeepLearningTrainingConfig, seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        generator=generator,
    )


def _prepare_targets(targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    targets = targets.float()
    if targets.ndim == 1 and outputs.ndim == 2 and outputs.shape[1] == 1:
        targets = targets.unsqueeze(1)
    if targets.shape != outputs.shape:
        targets = targets.reshape_as(outputs)

    return targets


def _copy_model_state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in copy.deepcopy(model.state_dict()).items()}


def _get_default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_checkpoint_path(output_dir: Path | None, model_name: str, seed: int) -> Path | None:
    if output_dir is None:
        return None

    return output_dir / f"{model_name}_seed_{seed}_best.pt"


def _get_mapping(config: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = config.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section must be a mapping: {key}")

    return value
