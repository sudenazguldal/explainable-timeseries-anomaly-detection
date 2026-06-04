from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CHERRY_RED = "#73070E"
DILL_GREEN = "#4E6813"

DATASET_COLORS = {
    "BATADAL": CHERRY_RED,
    "SKAB": DILL_GREEN,
}


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def load_fixed_original_results(summary_path: Path) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing file: {summary_path}. "
            "Run src.experiments.collect_automata_results first."
        )

    summary_df = pd.read_csv(summary_path)

    fixed_df = summary_df[
        (summary_df["scenario"] == "original")
        & (summary_df["evaluation_scope"] == "full_test_transitions_with_unseen_handling")
    ].copy()

    required_columns = ["dataset", "f1_score", "window_size", "alphabet_size"]

    missing_columns = [
        column
        for column in required_columns
        if column not in fixed_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns in fixed summary: {missing_columns}")

    return fixed_df


def load_best_parameter_results(best_path: Path) -> pd.DataFrame:
    if not best_path.exists():
        raise FileNotFoundError(
            f"Missing file: {best_path}. "
            "Run src.experiments.collect_best_parameter_results first."
        )

    best_df = pd.read_csv(best_path)

    required_columns = [
        "dataset",
        "best_window_size",
        "best_alphabet_size",
        "best_f1_score",
        "state_count",
        "transition_density",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in best_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns in best summary: {missing_columns}")

    return best_df


def create_fixed_vs_best_dataframe(
    fixed_df: pd.DataFrame,
    best_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    dataset_order = ["BATADAL", "SKAB"]

    for dataset in dataset_order:
        fixed_row = fixed_df[fixed_df["dataset"] == dataset]
        best_row = best_df[best_df["dataset"] == dataset]

        if fixed_row.empty:
            raise ValueError(f"Missing fixed original result for dataset={dataset}")

        if best_row.empty:
            raise ValueError(f"Missing best parameter result for dataset={dataset}")

        fixed_row = fixed_row.iloc[0]
        best_row = best_row.iloc[0]

        rows.append({
            "dataset": dataset,
            "setting": "Fixed\nw=4, a=3",
            "f1_score": float(fixed_row["f1_score"]),
            "window_size": int(fixed_row["window_size"]),
            "alphabet_size": int(fixed_row["alphabet_size"]),
        })

        rows.append({
            "dataset": dataset,
            "setting": (
                f"Best\nw={int(best_row['best_window_size'])}, "
                f"a={int(best_row['best_alphabet_size'])}"
            ),
            "f1_score": float(best_row["best_f1_score"]),
            "window_size": int(best_row["best_window_size"]),
            "alphabet_size": int(best_row["best_alphabet_size"]),
            "state_count": float(best_row["state_count"]),
            "transition_density": float(best_row["transition_density"]),
        })

    return pd.DataFrame(rows)


def plot_fixed_vs_best_f1(plot_df: pd.DataFrame, output_path: Path) -> None:
    datasets = ["BATADAL", "SKAB"]
    settings = ["Fixed\nw=4, a=3", "Best"]

    x_positions = list(range(len(datasets)))
    bar_width = 0.35

    plt.figure(figsize=(9, 6))

    for dataset_index, dataset in enumerate(datasets):
        dataset_df = plot_df[plot_df["dataset"] == dataset].copy()

        fixed_row = dataset_df[dataset_df["setting"].str.startswith("Fixed")].iloc[0]
        best_row = dataset_df[dataset_df["setting"].str.startswith("Best")].iloc[0]

        color = DATASET_COLORS[dataset]

        fixed_position = x_positions[dataset_index] - bar_width / 2
        best_position = x_positions[dataset_index] + bar_width / 2

        plt.bar(
            fixed_position,
            fixed_row["f1_score"],
            width=bar_width,
            color=color,
            alpha=0.45,
            label="Fixed parameters" if dataset_index == 0 else None,
        )

        plt.bar(
            best_position,
            best_row["f1_score"],
            width=bar_width,
            color=color,
            alpha=0.95,
            hatch="//",
            label="Best parameter sweep" if dataset_index == 0 else None,
        )

        plt.text(
            fixed_position,
            fixed_row["f1_score"] + 0.01,
            f"{fixed_row['f1_score']:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

        plt.text(
            best_position,
            best_row["f1_score"] + 0.01,
            f"{best_row['f1_score']:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(x_positions, datasets)
    plt.xlabel("Dataset")
    plt.ylabel("F1-score")
    plt.title("Fixed vs Best Automata Parameter F1-Score")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    results_dir = Path("reports/results")
    figures_dir = Path("reports/figures")

    ensure_output_dir(figures_dir)

    fixed_df = load_fixed_original_results(
        summary_path=results_dir / "automata_summary_results.csv",
    )

    best_df = load_best_parameter_results(
        best_path=results_dir / "automata_best_parameter_summary.csv",
    )

    plot_df = create_fixed_vs_best_dataframe(
        fixed_df=fixed_df,
        best_df=best_df,
    )

    output_path = figures_dir / "automata_fixed_vs_best_parameter_f1.png"

    plot_fixed_vs_best_f1(
        plot_df=plot_df,
        output_path=output_path,
    )

    print("Fixed vs best parameter F1 plot created.")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()