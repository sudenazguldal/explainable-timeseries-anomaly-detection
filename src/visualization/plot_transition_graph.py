import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from src.config.load_config import load_config
from src.visualization.plot_transition_heatmap import (
    build_batadal_automata,
    build_skab_automata,
)


CHERRY_RED = "#73070E"
DILL_GREEN = "#4E6813"

# Transition graph için biraz daha açık tonlar.
# Bar chart ve heatmap renklerini bozmadan sadece graph okunabilirliğini artırır.
GRAPH_COLORS = {
    "BATADAL": "#8E1B23",
    "SKAB": "#6B8428",
}


def extract_top_transitions(
    automata,
    top_n_edges: int = 15,
    min_probability: float = 0.0,
) -> list[dict]:
    edges = []

    for current_state, next_state_counts in automata.transition_counts.items():
        probabilities = automata.transition_probabilities.get(current_state, {})

        for next_state, count in next_state_counts.items():
            probability = float(probabilities.get(next_state, 0.0))

            if probability < min_probability:
                continue

            edges.append({
                "current_state": current_state,
                "next_state": next_state,
                "count": int(count),
                "probability": probability,
            })

    edges = sorted(
        edges,
        key=lambda edge: (edge["probability"], edge["count"]),
        reverse=True,
    )

    return edges[:top_n_edges]


def calculate_circular_positions(states: list[str]) -> dict[str, tuple[float, float]]:
    positions = {}
    n_states = len(states)

    for index, state in enumerate(states):
        angle = 2 * math.pi * index / n_states
        positions[state] = (
            math.cos(angle),
            math.sin(angle),
        )

    return positions


def write_top_transitions_csv(edges: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["current_state", "next_state", "count", "probability"]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edges)


def plot_transition_graph(
    edges: list[dict],
    output_path: Path,
    title: str,
    graph_color: str,
    label_probability_threshold: float = 0.70,
) -> None:
    states = sorted(
        set(edge["current_state"] for edge in edges)
        | set(edge["next_state"] for edge in edges)
    )

    positions = calculate_circular_positions(states)

    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # Nodes
    for state, (x, y) in positions.items():
        plt.scatter(
            x,
            y,
            s=900,
            color=graph_color,
            edgecolors="black",
            linewidths=1.2,
            zorder=3,
        )

        plt.text(
            x,
            y,
            state,
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
            zorder=4,
        )

    max_count = max(edge["count"] for edge in edges) if edges else 1

    # Edges
    for edge in edges:
        start = positions[edge["current_state"]]
        end = positions[edge["next_state"]]

        if edge["current_state"] == edge["next_state"]:
            continue

        width = 0.5 + 3.0 * (edge["count"] / max_count)

        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=12,
            linewidth=width,
            shrinkA=25,
            shrinkB=25,
            connectionstyle="arc3,rad=0.12",
            color=graph_color,
            alpha=0.85,
            zorder=2,
        )

        ax.add_patch(arrow)

        if edge["probability"] >= label_probability_threshold:
            label_x = (start[0] + end[0]) / 2
            label_y = (start[1] + end[1]) / 2

            plt.text(
                label_x,
                label_y,
                f"{edge['probability']:.2f}",
                fontsize=7,
                ha="center",
                va="center",
                color="black",
                bbox=dict(
                    facecolor="white",
                    alpha=0.70,
                    edgecolor="none",
                    pad=1,
                ),
                zorder=5,
            )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_graph_for_dataset(
    dataset_name: str,
    automata,
    config: dict,
    figure_path: Path,
    csv_path: Path,
) -> None:
    automata_config = config["automata"]

    window_size = automata_config["fixed"]["window_size"]
    alphabet_size = automata_config["fixed"]["alphabet_size"]

    edges = extract_top_transitions(
        automata=automata,
        top_n_edges=15,
        min_probability=0.0,
    )

    title = (
        f"{dataset_name} Original Automata State Transition Graph\n"
        f"top-15 transitions, window_size={window_size}, alphabet_size={alphabet_size}"
    )

    plot_transition_graph(
        edges=edges,
        output_path=figure_path,
        title=title,
        graph_color=GRAPH_COLORS[dataset_name],
        label_probability_threshold=0.70,
    )

    write_top_transitions_csv(
        edges=edges,
        output_path=csv_path,
    )


def main() -> None:
    config = load_config("config.yaml")

    batadal_automata = build_batadal_automata(config)
    skab_automata = build_skab_automata(config)

    create_graph_for_dataset(
        dataset_name="BATADAL",
        automata=batadal_automata,
        config=config,
        figure_path=Path("reports/figures/automata_transition_graph_batadal.png"),
        csv_path=Path("reports/tables/automata_top_transitions_batadal.csv"),
    )

    create_graph_for_dataset(
        dataset_name="SKAB",
        automata=skab_automata,
        config=config,
        figure_path=Path("reports/figures/automata_transition_graph_skab.png"),
        csv_path=Path("reports/tables/automata_top_transitions_skab.csv"),
    )

    print("Automata transition graphs created.")
    print(f"BATADAL state count: {batadal_automata.state_count()}")
    print(f"BATADAL transition density: {batadal_automata.transition_density()}")
    print(f"SKAB state count: {skab_automata.state_count()}")
    print(f"SKAB transition density: {skab_automata.transition_density()}")
    print("Created files:")
    print("- reports/figures/automata_transition_graph_batadal.png")
    print("- reports/figures/automata_transition_graph_skab.png")
    print("- reports/tables/automata_top_transitions_batadal.csv")
    print("- reports/tables/automata_top_transitions_skab.csv")


if __name__ == "__main__":
    main()