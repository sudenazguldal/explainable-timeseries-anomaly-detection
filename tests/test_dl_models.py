import torch

from src.models.cnn1d_model import CNN1DModel
from src.models.lstm_model import LSTMModel


def test_lstm_forward_pass_returns_binary_logit_shape():
    model = LSTMModel(input_size=3, hidden_size=4, output_size=1)
    inputs = torch.randn(2, 5, 3)

    outputs = model(inputs)

    assert outputs.shape == (2, 1)


def test_cnn1d_forward_pass_returns_binary_logit_shape():
    model = CNN1DModel(input_channels=3, hidden_channels=(4,), kernel_sizes=3, output_size=1)
    inputs = torch.randn(2, 3, 5)

    outputs = model(inputs)

    assert outputs.shape == (2, 1)
