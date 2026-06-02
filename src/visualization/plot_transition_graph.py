import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from src.visualization.plot_transition_heatmap import build_batadal_automata
from src.config.load_config import load_config


def extract_top_transitions(automata, top_n_edges: int = 20) -> list[dict]:
    """
    Extracts top transitions by transition count.

    Each edge contains:
    - current state
    - next state
    - transition count
    - transition probability
    """
    edges = []

    for current_state, next_state_counts in automata.transition_counts.items():
        probabilities = automata.transition_probabilities.get(current_state, {})

        for next_state, count in next_state_counts.items():
            edges.append({
                "current_state": current_state,
                "next_state": next_state,
                "count": int(count),
                "probability": float(probabilities.get(next_state, 0.0)),
            })

    edges = sorted(
        edges,
        key=lambda edge: (edge["count"], edge["probability"]),
        reverse=True,
    )

    return edges[:top_n_edges]


def calculate_circular_positions(states: list[str]) -> dict[str, tuple[float, float]]:
    """
    Places states on a circle for a readable directed graph.
    """
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


def plot_transition_graph(edges: list[dict], output_path: Path) -> None:
    """
    Plots a directed state transition graph.

    To keep the graph readable, only top transitions are drawn.
    """
    states = sorted(
        set(edge["current_state"] for edge in edges)
        | set(edge["next_state"] for edge in edges)
    )

    positions = calculate_circular_positions(states)

    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # Draw nodes
    for state, (x, y) in positions.items():
        plt.scatter(x, y, s=900)
        plt.text(
            x,
            y,
            state,
            ha="center",
            va="center",
            fontsize=8,
        )

    max_count = max(edge["count"] for edge in edges) if edges else 1

    # Draw directed edges
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
        )
        ax.add_patch(arrow)

        label_x = (start[0] + end[0]) / 2
        label_y = (start[1] + end[1]) / 2

        plt.text(
            label_x,
            label_y,
            f"{edge['probability']:.2f}",
            fontsize=7,
            ha="center",
            va="center",
        )

    plt.title("BATADAL Automata State Transition Graph")
    plt.axis("off")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    config = load_config("config.yaml")

    automata = build_batadal_automata(config)

    edges = extract_top_transitions(
        automata=automata,
        top_n_edges=20,
    )

    figure_path = Path("reports/figures/automata_transition_graph_batadal.png")
    csv_path = Path("reports/tables/automata_top_transitions_batadal.csv")

    plot_transition_graph(
        edges=edges,
        output_path=figure_path,
    )

    write_top_transitions_csv(
        edges=edges,
        output_path=csv_path,
    )

    print("Automata transition graph created.")
    print(f"State count: {automata.state_count()}")
    print(f"Transition density: {automata.transition_density()}")
    print(f"Figure path: {figure_path}")
    print(f"Top transition table path: {csv_path}")


if __name__ == "__main__":
    main()