from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config.load_config import load_config
from src.data.batadal_loader import (
    load_batadal_dataset,
    detect_batadal_target_column,
    detect_batadal_time_column,
    get_batadal_feature_columns,
    split_batadal_time_ordered,
)
from src.data.label_utils import normalize_batadal_labels
from src.preprocessing.scaler import fit_transform_train
from src.preprocessing.pca import fit_transform_train_pca
from src.models.automata.paa import paa_transform
from src.models.automata.sax import sax_transform
from src.models.automata.probabilistic_automata import ProbabilisticAutomata


def series_to_symbol_sequence(values, n_segments: int, alphabet_size: int) -> str:
    paa_values = paa_transform(values, n_segments=n_segments)
    return sax_transform(paa_values, alphabet_size=alphabet_size)


def build_batadal_automata(config: dict) -> ProbabilisticAutomata:
    batadal_config = config["datasets"]["batadal"]
    preprocessing_config = config["preprocessing"]
    automata_config = config["automata"]

    df = load_batadal_dataset(
        raw_path=batadal_config["raw_path"],
    )

    target_column = detect_batadal_target_column(
        df=df,
        target_column_candidates=batadal_config["target_column_candidates"],
    )

    time_column = detect_batadal_time_column(
        df=df,
        time_column_candidates=batadal_config["time_column_candidates"],
    )

    df = normalize_batadal_labels(
        df=df,
        target_column=target_column,
        output_column="label",
    )

    feature_columns = get_batadal_feature_columns(
        df=df,
        target_column=target_column,
        time_column=time_column,
    )

    split_config = batadal_config["split"]

    train_df, _, _ = split_batadal_time_ordered(
        df=df,
        train_ratio=split_config["train"],
        validation_ratio=split_config["validation"],
        test_ratio=split_config["test"],
        time_column=time_column,
    )

    scaled_train_df, _ = fit_transform_train(
        train_df=train_df,
        feature_columns=feature_columns,
        method=preprocessing_config["normalization"],
    )

    pca_train_df, _ = fit_transform_train_pca(
        train_df=scaled_train_df,
        feature_columns=feature_columns,
        n_components=preprocessing_config["pca_components"],
        output_column="pc1",
    )

    window_size = automata_config["fixed"]["window_size"]
    alphabet_size = automata_config["fixed"]["alphabet_size"]
    paa_segments = min(
        automata_config.get("paa_segments", 256),
        len(pca_train_df),
    )

    train_symbol_sequence = series_to_symbol_sequence(
        values=pca_train_df["pc1"].to_numpy(),
        n_segments=paa_segments,
        alphabet_size=alphabet_size,
    )

    automata = ProbabilisticAutomata(
        window_size=window_size,
        fallback_probability=automata_config.get(
            "fallback_probability",
            automata_config.get("smoothing", 0.000001),
        ),
    )

    automata.fit(train_symbol_sequence)

    return automata


def transition_probabilities_to_matrix(
    automata: ProbabilisticAutomata,
    top_n_states: int = 25,
) -> pd.DataFrame:
    """
    Converts transition probabilities into a square matrix.

    To keep the heatmap readable, only the states with the highest outgoing transition counts are selected.
    """
    state_activity = []

    for state in automata.states:
        outgoing_count = sum(
            automata.transition_counts.get(state, {}).values()
        )
        state_activity.append((state, outgoing_count))

    selected_states = [
        state
        for state, _ in sorted(
            state_activity,
            key=lambda item: item[1],
            reverse=True,
        )[:top_n_states]
    ]

    matrix = pd.DataFrame(
        0.0,
        index=selected_states,
        columns=selected_states,
    )

    for current_state in selected_states:
        next_states = automata.transition_probabilities.get(current_state, {})

        for next_state, probability in next_states.items():
            if next_state in matrix.columns:
                matrix.loc[current_state, next_state] = float(probability)

    return matrix


def plot_heatmap(matrix: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 10))

    image = plt.imshow(matrix.values, aspect="auto")
    plt.colorbar(image, label="Transition probability")

    plt.xticks(
        ticks=range(len(matrix.columns)),
        labels=matrix.columns,
        rotation=90,
        fontsize=7,
    )
    plt.yticks(
        ticks=range(len(matrix.index)),
        labels=matrix.index,
        fontsize=7,
    )

    plt.title("BATADAL Automata Transition Probability Heatmap")
    plt.xlabel("Next state")
    plt.ylabel("Current state")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    config = load_config("config.yaml")

    automata = build_batadal_automata(config)

    matrix = transition_probabilities_to_matrix(
        automata=automata,
        top_n_states=25,
    )

    output_path = Path("reports/figures/automata_transition_heatmap_batadal.png")

    plot_heatmap(
        matrix=matrix,
        output_path=output_path,
    )

    print("Transition probability heatmap created.")
    print(f"State count: {automata.state_count()}")
    print(f"Transition density: {automata.transition_density()}")
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    main()