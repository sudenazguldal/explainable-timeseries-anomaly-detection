from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_scenario_f1_comparison(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Creates a grouped bar chart for scenario-based F1-score comparison
    using automata_summary_results.csv.
    """
    df = summary_df.copy()

    # Keep a stable order for report readability
    scenario_order = ["original", "gaussian_noise", "unseen"]
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

    pivot_df = plot_df.pivot(index="scenario", columns="dataset", values="f1_score")
    pivot_df = pivot_df.reindex(scenario_order)

    ax = pivot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Automata F1-Score by Scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("F1-score")
    ax.legend(title="Dataset")
    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = output_dir / "automata_scenario_f1_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_scenario_unseen_ratio_comparison(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Creates a grouped bar chart for unseen ratio comparison
    using automata_summary_results.csv.
    """
    df = summary_df.copy()

    scenario_order = ["original", "gaussian_noise", "unseen"]
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

    pivot_df = plot_df.pivot(index="scenario", columns="dataset", values="unseen_ratio")
    pivot_df = pivot_df.reindex(scenario_order)

    ax = pivot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Automata Unseen Ratio by Scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Unseen ratio")
    ax.legend(title="Dataset")
    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = output_dir / "automata_scenario_unseen_ratio_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_multiseed_f1_errorbars(multiseed_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Creates a mean/std error-bar chart for F1-score
    using automata_multiseed_summary.csv.
    """
    df = multiseed_df.copy()

    labels = [
        f"{row['dataset']} | {row['scenario']}"
        for _, row in df.iterrows()
    ]
    means = df["f1_score_mean"].astype(float).tolist()
    stds = df["f1_score_std"].astype(float).tolist()

    positions = list(range(len(labels)))

    plt.figure(figsize=(12, 6))
    plt.bar(positions, means, yerr=stds, capsize=5)
    plt.xticks(positions, labels, rotation=30, ha="right")
    plt.ylabel("F1-score")
    plt.title("Automata Multi-Seed F1-Score Mean ± Std")
    plt.tight_layout()

    output_path = output_dir / "automata_multiseed_f1_errorbars.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_multiseed_accuracy_errorbars(multiseed_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Creates a mean/std error-bar chart for accuracy
    using automata_multiseed_summary.csv.
    """
    df = multiseed_df.copy()

    labels = [
        f"{row['dataset']} | {row['scenario']}"
        for _, row in df.iterrows()
    ]
    means = df["accuracy_mean"].astype(float).tolist()
    stds = df["accuracy_std"].astype(float).tolist()

    positions = list(range(len(labels)))

    plt.figure(figsize=(12, 6))
    plt.bar(positions, means, yerr=stds, capsize=5)
    plt.xticks(positions, labels, rotation=30, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Automata Multi-Seed Accuracy Mean ± Std")
    plt.tight_layout()

    output_path = output_dir / "automata_multiseed_accuracy_errorbars.png"
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

    plot_scenario_f1_comparison(summary_df, figures_dir)
    plot_scenario_unseen_ratio_comparison(summary_df, figures_dir)
    plot_multiseed_f1_errorbars(multiseed_df, figures_dir)
    plot_multiseed_accuracy_errorbars(multiseed_df, figures_dir)

    print("Automata result plots created.")
    print(f"Output directory: {figures_dir}")
    print("Created files:")
    print("- automata_scenario_f1_comparison.png")
    print("- automata_scenario_unseen_ratio_comparison.png")
    print("- automata_multiseed_f1_errorbars.png")
    print("- automata_multiseed_accuracy_errorbars.png")


if __name__ == "__main__":
    main()