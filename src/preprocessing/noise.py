import numpy as np
import pandas as pd


def add_gaussian_noise(
    df: pd.DataFrame,
    feature_columns: list[str],
    mean: float = 0.0,
    std: float = 0.05,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Adds Gaussian noise to selected feature columns.  Used for noise robustness experiments.
    """
    noisy_df = df.copy()

    rng = np.random.default_rng(random_seed)

    noise = rng.normal(
        loc=mean,
        scale=std,
        size=noisy_df[feature_columns].shape,
    )

    noisy_df[feature_columns] = noisy_df[feature_columns] + noise

    return noisy_df