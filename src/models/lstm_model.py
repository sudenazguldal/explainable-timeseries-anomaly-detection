from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class LSTMModelConfig:
    """
    Configuration for an LSTM time-series baseline.
    """

    input_size: int
    hidden_size: int = 64
    num_layers: int = 1
    output_size: int = 1
    dropout: float = 0.0
    bidirectional: bool = False
    batch_first: bool = True


class LSTMModel(nn.Module):
    """
    Modular LSTM baseline for sequence classification or regression.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_size: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        batch_first: bool = True,
    ):
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be greater than zero.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero.")
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero.")
        if output_size <= 0:
            raise ValueError("output_size must be greater than zero.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        direction_count = 2 if bidirectional else 1
        head_input_size = hidden_size * direction_count

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_size),
            nn.Dropout(dropout),
            nn.Linear(head_input_size, output_size),
        )

    @classmethod
    def from_config(cls, config: LSTMModelConfig | Mapping[str, object]) -> "LSTMModel":
        """
        Creates an LSTM model from a dataclass or dictionary configuration.
        """
        if isinstance(config, LSTMModelConfig):
            return cls(**config.__dict__)

        if "input_size" not in config:
            raise ValueError("input_size is required for LSTMModel.")

        return cls(
            input_size=int(config["input_size"]),
            hidden_size=int(config.get("hidden_size", 64)),
            num_layers=int(config.get("num_layers", 1)),
            output_size=int(config.get("output_size", 1)),
            dropout=float(config.get("dropout", 0.0)),
            bidirectional=bool(config.get("bidirectional", False)),
            batch_first=bool(config.get("batch_first", True)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Runs a forward pass over input windows.
        """
        if inputs.ndim != 3:
            raise ValueError("LSTM inputs must have shape (batch, sequence, features).")
        if inputs.shape[-1] != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, received {inputs.shape[-1]}.")

        outputs, _ = self.lstm(inputs)
        final_output = outputs[:, -1, :] if self.batch_first else outputs[-1, :, :]

        return self.head(final_output)


def create_lstm_model_from_config(config: Mapping[str, object], input_size: int, output_size: int = 1) -> LSTMModel:
    """
    Creates an LSTM baseline using a project-level model configuration.
    """
    model_config = dict(config)
    model_config["input_size"] = input_size
    model_config.setdefault("output_size", output_size)

    return LSTMModel.from_config(model_config)


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Counts trainable model parameters.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
