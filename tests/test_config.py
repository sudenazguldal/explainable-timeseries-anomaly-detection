from src.config.load_config import load_config


def test_config_loads_successfully():
    config = load_config("config.yaml")

    assert "datasets" in config
    assert "automata" in config
    assert "deep_learning" in config
    assert "experiments" in config


def test_required_random_seeds_exist():
    config = load_config("config.yaml")

    assert config["project"]["random_seeds"] == [42, 123, 2026, 7, 999]


def test_deep_learning_models_are_lstm_and_cnn1d():
    config = load_config("config.yaml")

    assert "lstm" in config["deep_learning"]["models"]
    assert "cnn1d" in config["deep_learning"]["models"]
    

def test_automata_probability_parameters_are_numeric():
    config = load_config("config.yaml")

    assert isinstance(config["automata"]["fallback_probability"], float)
    assert isinstance(config["automata"]["anomaly_threshold"], float)
    assert config["automata"]["fallback_probability"] > 0
    assert 0 < config["automata"]["anomaly_threshold"] < 1