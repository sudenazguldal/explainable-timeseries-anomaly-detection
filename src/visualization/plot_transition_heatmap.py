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
from src.data.skab_loader import load_skab_dataset, get_skab_feature_columns
from src.data.label_utils import normalize_batadal_labels, normalize_skab_labels
from src.data.splitter import create_skab_group_folds
from src.preprocessing.scaler import fit_transform_train
from src.preprocessing.pca import fit_transform_train_pca
from src.models.automata.paa import paa_transform
from src.models.automata.sax import sax_transform
from src.models.automata.probabilistic_automata import ProbabilisticAutomata


def series_to_symbol_sequence(values, n_segments: int, alphabet_size: int) -> str:
    paa_values = paa_transform(values, n_segments=n_segments)
    return sax_transform(paa_values, alphabet_size=alphabet_size)


def get_fallback_probability(automata_config: dict) -> float:
    return float(automata_config.get("fallback_probability", 0.000001))


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
        fallback_probability=get_fallback_probability(automata_config),
    )

    automata.fit(train_symbol_sequence)

    return automata


def build_skab_automata(config: dict) -> ProbabilisticAutomata:
    skab_config = config["datasets"]["skab"]
    preprocessing_config = config["preprocessing"]
    automata_config = config["automata"]

    df = load_skab_dataset(
        raw_path=skab_config["raw_path"],
        use_groups=skab_config["use_groups"],
    )

    df = normalize_skab_labels(
        df=df,
        target_column=skab_config["target_column"],
        output_column="label",
    )

    feature_columns = get_skab_feature_columns(
        df=df,
        target_column=skab_config["target_column"],
        excluded_columns=skab_config["excluded_columns"],
    )

    folds = create_skab_group_folds(
        df=df,
        target_column="label",
        group_column=skab_config["group_column"],
        n_splits=5,
        stratified=True,
        random_seed=config["project"]["random_seeds"][0],
    )

    train_indices, _ = folds[0]
    train_df = df.iloc[train_indices].copy()

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
        fallback_probability=get_fallback_probability(automata_config),
    )

    automata.fit(train_symbol_sequence)

    return automata


def transition_probabilities_to_matrix(
    automata: ProbabilisticAutomata,
    top_n_states: int = 25,
) -> pd.DataFrame:
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


def plot_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
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

    plt.title(title)
    plt.xlabel("Next state")
    plt.ylabel("Current state")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_heatmap_for_dataset(
    dataset_name: str,
    automata: ProbabilisticAutomata,
    config: dict,
    output_path: Path,
) -> None:
    automata_config = config["automata"]

    window_size = automata_config["fixed"]["window_size"]
    alphabet_size = automata_config["fixed"]["alphabet_size"]

    matrix = transition_probabilities_to_matrix(
        automata=automata,
        top_n_states=25,
    )

    title = (
        f"{dataset_name} Original Automata Transition Probability Heatmap\n"
        f"window_size={window_size}, alphabet_size={alphabet_size}, top-25 active states"
    )

    plot_heatmap(
        matrix=matrix,
        output_path=output_path,
        title=title,
    )


def main() -> None:
    config = load_config("config.yaml")

    batadal_automata = build_batadal_automata(config)
    skab_automata = build_skab_automata(config)

    create_heatmap_for_dataset(
        dataset_name="BATADAL",
        automata=batadal_automata,
        config=config,
        output_path=Path("reports/figures/automata_transition_heatmap_batadal.png"),
    )

    create_heatmap_for_dataset(
        dataset_name="SKAB",
        automata=skab_automata,
        config=config,
        output_path=Path("reports/figures/automata_transition_heatmap_skab.png"),
    )

    print("Transition probability heatmaps created.")
    print(f"BATADAL state count: {batadal_automata.state_count()}")
    print(f"BATADAL transition density: {batadal_automata.transition_density()}")
    print(f"SKAB state count: {skab_automata.state_count()}")
    print(f"SKAB transition density: {skab_automata.transition_density()}")
    print("Created files:")
    print("- reports/figures/automata_transition_heatmap_batadal.png")
    print("- reports/figures/automata_transition_heatmap_skab.png")


if __name__ == "__main__":
    main()