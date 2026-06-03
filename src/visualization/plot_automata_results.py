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


def plot_scenario_f1_comparison(summary_df: pd.DataFrame, output_dir: Path) -> None:
    df = summary_df.copy()

    scenario_order = ["original", "gaussian_noise", "unseen_only"]
    scenario_labels = {
        "original": "Original\nfull test",
        "gaussian_noise": "Gaussian noise\nfull test",
        "unseen_only": "Unseen-only\nsubset",
    }
    dataset_order = ["BATADAL", "SKAB"]

    rows = []

    for dataset in dataset_order:
        dataset_df = df[df["dataset"] == dataset].copy()

        for scenario in scenario_order:
            row = dataset_df[dataset_df["scenario"] == scenario]

            if not row.empty:
                rows.append({
                    "dataset": dataset,
                    "scenario": scenario,
                    "f1_score": float(row.iloc[0]["f1_score"]),
                })

    plot_df = pd.DataFrame(rows)

    pivot_df = plot_df.pivot(
        index="scenario",
        columns="dataset",
        values="f1_score",
    )

    pivot_df = pivot_df.reindex(scenario_order)

    ax = pivot_df.plot(
        kind="bar",
        figsize=(10, 6),
        color=[DATASET_COLORS["BATADAL"], DATASET_COLORS["SKAB"]],
    )
    ax.set_title("Automata F1-Score by Evaluation Scope")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("F1-score")
    ax.legend(title="Dataset")

    ax.set_xticklabels(
        [scenario_labels[scenario] for scenario in scenario_order],
        rotation=0,
    )
    plt.tight_layout()

    output_path = output_dir / "automata_scenario_f1_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_scenario_unseen_ratio_comparison(summary_df: pd.DataFrame, output_dir: Path) -> None:
    df = summary_df.copy()

    scenario_order = ["original", "gaussian_noise", "unseen_only"]
    scenario_labels = {
        "original": "Original\nfull test",
        "gaussian_noise": "Gaussian noise\nfull test",
        "unseen_only": "Unseen-only\nsubset",
    }
    dataset_order = ["BATADAL", "SKAB"]

    rows = []

    for dataset in dataset_order:
        dataset_df = df[df["dataset"] == dataset].copy()

        for scenario in scenario_order:
            row = dataset_df[dataset_df["scenario"] == scenario]

            if not row.empty:
                rows.append({
                    "dataset": dataset,
                    "scenario": scenario,
                    "unseen_ratio": float(row.iloc[0]["unseen_ratio"]),
                })

    plot_df = pd.DataFrame(rows)

    pivot_df = plot_df.pivot(
        index="scenario",
        columns="dataset",
        values="unseen_ratio",
    )

    pivot_df = pivot_df.reindex(scenario_order)

    ax = pivot_df.plot(
        kind="bar",
        figsize=(10, 6),
        color=[DATASET_COLORS["BATADAL"], DATASET_COLORS["SKAB"]],
    )
    ax.set_title("Observed Unseen Pattern Ratio by Evaluation Scope")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Observed unseen pattern ratio")
    ax.legend(title="Dataset")

    ax.set_xticklabels(
        [scenario_labels[scenario] for scenario in scenario_order],
        rotation=0,
    )
    plt.tight_layout()

    output_path = output_dir / "automata_scenario_unseen_ratio_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_multiseed_metric_errorbars(
    multiseed_df: pd.DataFrame,
    output_dir: Path,
    metric_name: str,
    ylabel: str,
    title: str,
    output_filename: str,
) -> None:
    df = multiseed_df.copy()

    scenario_order = ["original", "gaussian_noise", "unseen_only"]
    scenario_labels = {
        "original": "Original\nfull test",
        "gaussian_noise": "Gaussian noise\nfull test",
        "unseen_only": "Unseen-only\nsubset",
    }
    dataset_order = ["BATADAL", "SKAB"]

    mean_column = f"{metric_name}_mean"
    std_column = f"{metric_name}_std"

    x_positions = list(range(len(scenario_order)))
    bar_width = 0.35

    plt.figure(figsize=(10, 6))

    for dataset_index, dataset in enumerate(dataset_order):
        means = []
        stds = []

        for scenario in scenario_order:
            row = df[
                (df["dataset"] == dataset)
                & (df["scenario"] == scenario)
            ]

            if row.empty:
                raise ValueError(
                    f"Missing multi-seed result for dataset={dataset}, scenario={scenario}"
                )
            else:
                means.append(float(row.iloc[0][mean_column]))
                stds.append(float(row.iloc[0][std_column]))

        offset = (dataset_index - 0.5) * bar_width
        positions = [
            position + offset
            for position in x_positions
        ]

        plt.bar(
            positions,
            means,
            width=bar_width,
            yerr=stds,
            capsize=5,
            label=dataset,
            color=DATASET_COLORS[dataset],
        )

    plt.xticks(
        x_positions,
        [scenario_labels[scenario] for scenario in scenario_order],
    )
    plt.xlabel("Scenario")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Dataset")
    plt.tight_layout()

    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    results_dir = Path("reports/results")
    figures_dir = Path("reports/figures")

    ensure_output_dir(figures_dir)

    summary_results_path = results_dir / "automata_summary_results.csv"
    multiseed_summary_path = results_dir / "automata_multiseed_summary.csv"

    summary_df = pd.read_csv(summary_results_path)
    multiseed_df = pd.read_csv(multiseed_summary_path)

    plot_scenario_f1_comparison(
        summary_df=summary_df,
        output_dir=figures_dir,
    )

    plot_scenario_unseen_ratio_comparison(
        summary_df=summary_df,
        output_dir=figures_dir,
    )

    plot_multiseed_metric_errorbars(
        multiseed_df=multiseed_df,
        output_dir=figures_dir,
        metric_name="f1_score",
        ylabel="F1-score",
        title="Automata Multi-Seed F1-Score Mean ± Std",
        output_filename="automata_multiseed_f1_errorbars.png",
    )

    plot_multiseed_metric_errorbars(
        multiseed_df=multiseed_df,
        output_dir=figures_dir,
        metric_name="accuracy",
        ylabel="Accuracy",
        title="Automata Multi-Seed Accuracy Mean ± Std",
        output_filename="automata_multiseed_accuracy_errorbars.png",
    )

    print("Automata result plots created.")
    print(f"Output directory: {figures_dir}")
    print("Created files:")
    print("- automata_scenario_f1_comparison.png")
    print("- automata_scenario_unseen_ratio_comparison.png")
    print("- automata_multiseed_f1_errorbars.png")
    print("- automata_multiseed_accuracy_errorbars.png")


if __name__ == "__main__":
    main()