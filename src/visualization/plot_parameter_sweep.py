from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def normalize_parameter_sweep_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes BATADAL and SKAB parameter sweep CSV column names.

    BATADAL columns:
    - f1_score
    - state_count
    - transition_density

    SKAB columns:
    - f1_score_mean
    - state_count_mean
    - transition_density_mean
    """
    normalized_df = df.copy()

    if "f1_score" not in normalized_df.columns and "f1_score_mean" in normalized_df.columns:
        normalized_df["f1_score"] = normalized_df["f1_score_mean"]

    if "state_count" not in normalized_df.columns and "state_count_mean" in normalized_df.columns:
        normalized_df["state_count"] = normalized_df["state_count_mean"]

    if (
        "transition_density" not in normalized_df.columns
        and "transition_density_mean" in normalized_df.columns
    ):
        normalized_df["transition_density"] = normalized_df["transition_density_mean"]

    required_columns = [
        "window_size",
        "alphabet_size",
        "f1_score",
        "state_count",
        "transition_density",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in normalized_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return normalized_df


def create_heatmap_matrix(
    df: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """
    Creates a window_size x alphabet_size matrix for heatmap plotting.
    """
    matrix = df.pivot(
        index="window_size",
        columns="alphabet_size",
        values=value_column,
    )

    matrix = matrix.sort_index()
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)

    return matrix


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    colorbar_label: str,
    output_path: Path,
    value_format: str,
) -> None:
    """
    Plots a parameter sweep heatmap.

    Rows: window size
    Columns: alphabet size
    Cell value: selected metric
    """
    plt.figure(figsize=(8, 6))

    image = plt.imshow(matrix.values, aspect="auto")
    plt.colorbar(image, label=colorbar_label)

    plt.xticks(
        ticks=range(len(matrix.columns)),
        labels=matrix.columns,
    )
    plt.yticks(
        ticks=range(len(matrix.index)),
        labels=matrix.index,
    )

    for row_index in range(len(matrix.index)):
        for column_index in range(len(matrix.columns)):
            value = matrix.values[row_index, column_index]
            plt.text(
                column_index,
                row_index,
                format(value, value_format),
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.title(title)
    plt.xlabel("Alphabet size")
    plt.ylabel("Window size")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_dataset_parameter_heatmaps(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
) -> None:
    """
    Creates F1-score, state count and transition density heatmaps for one dataset.
    """
    normalized_df = normalize_parameter_sweep_columns(df)

    dataset_slug = dataset_name.lower()

    f1_matrix = create_heatmap_matrix(
        df=normalized_df,
        value_column="f1_score",
    )

    state_count_matrix = create_heatmap_matrix(
        df=normalized_df,
        value_column="state_count",
    )

    transition_density_matrix = create_heatmap_matrix(
        df=normalized_df,
        value_column="transition_density",
    )

    plot_heatmap(
        matrix=f1_matrix,
        title=f"{dataset_name} Automata Parameter Sweep: F1-Score",
        colorbar_label="F1-score",
        output_path=output_dir / f"automata_parameter_f1_heatmap_{dataset_slug}.png",
        value_format=".3f",
    )

    plot_heatmap(
        matrix=state_count_matrix,
        title=f"{dataset_name} Automata Parameter Sweep: State Count",
        colorbar_label="State count",
        output_path=output_dir / f"automata_parameter_state_count_heatmap_{dataset_slug}.png",
        value_format=".0f",
    )

    plot_heatmap(
        matrix=transition_density_matrix,
        title=f"{dataset_name} Automata Parameter Sweep: Transition Density",
        colorbar_label="Transition density",
        output_path=output_dir / f"automata_parameter_transition_density_heatmap_{dataset_slug}.png",
        value_format=".4f",
    )


def main() -> None:
    results_dir = Path("reports/results")
    figures_dir = Path("reports/figures")

    ensure_output_dir(figures_dir)

    batadal_path = results_dir / "automata_parameter_sweep_batadal.csv"
    skab_path = results_dir / "automata_parameter_sweep_skab.csv"

    if not batadal_path.exists():
        raise FileNotFoundError(
            f"Missing file: {batadal_path}. "
            "Run src.experiments.run_batadal_automata_parameter_sweep first."
        )

    if not skab_path.exists():
        raise FileNotFoundError(
            f"Missing file: {skab_path}. "
            "Run src.experiments.run_skab_automata_parameter_sweep first."
        )

    batadal_df = pd.read_csv(batadal_path)
    skab_df = pd.read_csv(skab_path)

    plot_dataset_parameter_heatmaps(
        df=batadal_df,
        dataset_name="BATADAL",
        output_dir=figures_dir,
    )

    plot_dataset_parameter_heatmaps(
        df=skab_df,
        dataset_name="SKAB",
        output_dir=figures_dir,
    )

    print("Automata parameter sweep heatmaps created.")
    print(f"Output directory: {figures_dir}")
    print("Created files:")
    print("- automata_parameter_f1_heatmap_batadal.png")
    print("- automata_parameter_f1_heatmap_skab.png")
    print("- automata_parameter_state_count_heatmap_batadal.png")
    print("- automata_parameter_state_count_heatmap_skab.png")
    print("- automata_parameter_transition_density_heatmap_batadal.png")
    print("- automata_parameter_transition_density_heatmap_skab.png")


if __name__ == "__main__":
    main()