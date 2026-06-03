from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class CNN1DModelConfig:
    """
    Configuration for a 1D-CNN time-series baseline.
    """

    input_channels: int
    output_size: int = 1
    hidden_channels: tuple[int, ...] = (32, 64)
    kernel_sizes: tuple[int, ...] = (5, 3)
    dropout: float = 0.1
    use_batch_norm: bool = True
    channels_first: bool = True


class CNN1DModel(nn.Module):
    """
    Modular 1D-CNN baseline for time-series classification or regression.
    """

    def __init__(
        self,
        input_channels: int,
        output_size: int = 1,
        hidden_channels: Sequence[int] = (32, 64),
        kernel_sizes: int | Sequence[int] = (5, 3),
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        channels_first: bool = True,
    ):
        super().__init__()

        if input_channels <= 0:
            raise ValueError("input_channels must be greater than zero.")
        if output_size <= 0:
            raise ValueError("output_size must be greater than zero.")
        if not hidden_channels:
            raise ValueError("hidden_channels must contain at least one layer size.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        self.input_channels = input_channels
        self.output_size = output_size
        self.hidden_channels = tuple(int(channel_count) for channel_count in hidden_channels)
        self.kernel_sizes = _expand_kernel_sizes(kernel_sizes, len(self.hidden_channels))
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.channels_first = channels_first

        self.feature_extractor = self._build_feature_extractor()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_channels[-1], output_size),
        )

    @classmethod
    def from_config(cls, config: CNN1DModelConfig | Mapping[str, object]) -> "CNN1DModel":
        """
        Creates a 1D-CNN model from a dataclass or dictionary configuration.
        """
        if isinstance(config, CNN1DModelConfig):
            return cls(**config.__dict__)

        if "input_channels" not in config:
            raise ValueError("input_channels is required for CNN1DModel.")

        return cls(
            input_channels=int(config["input_channels"]),
            output_size=int(config.get("output_size", 1)),
            hidden_channels=_as_int_tuple(config.get("hidden_channels", (32, 64))),
            kernel_sizes=_as_int_tuple(config.get("kernel_sizes", (5, 3))),
            dropout=float(config.get("dropout", 0.1)),
            use_batch_norm=bool(config.get("use_batch_norm", True)),
            channels_first=bool(config.get("channels_first", True)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Runs a forward pass over input windows.
        """
        if inputs.ndim != 3:
            raise ValueError("CNN1D inputs must be three-dimensional.")

        if self.channels_first:
            if inputs.shape[1] != self.input_channels:
                raise ValueError(f"Expected {self.input_channels} channels, received {inputs.shape[1]}.")
            model_inputs = inputs
        else:
            if inputs.shape[-1] != self.input_channels:
                raise ValueError(f"Expected {self.input_channels} channels, received {inputs.shape[-1]}.")
            model_inputs = inputs.transpose(1, 2)

        features = self.feature_extractor(model_inputs)
        pooled_features = self.global_pool(features).squeeze(-1)

        return self.head(pooled_features)

    def _build_feature_extractor(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        current_channels = self.input_channels

        for output_channels, kernel_size in zip(self.hidden_channels, self.kernel_sizes):
            if output_channels <= 0:
                raise ValueError("All hidden channel sizes must be greater than zero.")
            if kernel_size <= 0:
                raise ValueError("All kernel sizes must be greater than zero.")

            padding = kernel_size // 2
            layers.append(nn.Conv1d(current_channels, output_channels, kernel_size=kernel_size, padding=padding))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(output_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            current_channels = output_channels

        return nn.Sequential(*layers)


def create_cnn1d_model_from_config(config: Mapping[str, object], input_channels: int, output_size: int = 1) -> CNN1DModel:
    """
    Creates a 1D-CNN baseline using a project-level model configuration.
    """
    model_config = dict(config)
    model_config["input_channels"] = input_channels
    model_config.setdefault("output_size", output_size)

    return CNN1DModel.from_config(model_config)


def _expand_kernel_sizes(kernel_sizes: int | Sequence[int], layer_count: int) -> tuple[int, ...]:
    values = _as_int_tuple(kernel_sizes)
    if len(values) == 1:
        return values * layer_count
    if len(values) != layer_count:
        raise ValueError("kernel_sizes must have length 1 or match hidden_channels length.")

    return values


def _as_int_tuple(values: object) -> tuple[int, ...]:
    if isinstance(values, int):
        return (values,)
    if isinstance(values, str):
        return (int(values),)
    if isinstance(values, Sequence):
        return tuple(int(value) for value in values)

    raise ValueError("Expected an integer or sequence of integers.")
