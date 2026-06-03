import subprocess
import sys
from dataclasses import dataclass


@dataclass
class PipelineStep:
    name: str
    module: str


PIPELINE_STEPS = [
    PipelineStep(
        name="Run BATADAL original automata metrics",
        module="src.experiments.run_batadal_automata_metrics",
    ),
    PipelineStep(
        name="Run SKAB original automata metrics",
        module="src.experiments.run_skab_automata_metrics",
    ),
    PipelineStep(
        name="Run BATADAL Gaussian noise automata metrics",
        module="src.experiments.run_batadal_automata_noise_metrics",
    ),
    PipelineStep(
        name="Run SKAB Gaussian noise automata metrics",
        module="src.experiments.run_skab_automata_noise_metrics",
    ),
    PipelineStep(
        name="Run BATADAL unseen automata metrics",
        module="src.experiments.run_batadal_automata_unseen_metrics",
    ),
    PipelineStep(
        name="Run SKAB unseen automata metrics",
        module="src.experiments.run_skab_automata_unseen_metrics",
    ),
    PipelineStep(
        name="Collect automata scenario results",
        module="src.experiments.collect_automata_results",
    ),
    PipelineStep(
        name="Run automata multi-seed summary",
        module="src.experiments.run_automata_multiseed_summary",
    ),
    PipelineStep(
        name="Run BATADAL parameter sweep",
        module="src.experiments.run_batadal_automata_parameter_sweep",
    ),
    PipelineStep(
        name="Run SKAB parameter sweep",
        module="src.experiments.run_skab_automata_parameter_sweep",
    ),
    PipelineStep(
        name="Plot automata result charts",
        module="src.visualization.plot_automata_results",
    ),
    PipelineStep(
        name="Plot transition probability heatmaps",
        module="src.visualization.plot_transition_heatmap",
    ),
    PipelineStep(
        name="Plot transition graphs",
        module="src.visualization.plot_transition_graph",
    ),
    PipelineStep(
        name="Plot parameter sweep heatmaps",
        module="src.visualization.plot_parameter_sweep",
    ),
    PipelineStep(
        name="Plot precision-recall curves",
        module="src.visualization.plot_pr_curves",
    ),
]


def run_module(module_name: str) -> None:
    command = [
        sys.executable,
        "-m",
        module_name,
    ]

    subprocess.run(
        command,
        check=True,
    )


def run_tests() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
        ],
        check=True,
    )


def main() -> None:
    print("Full automata pipeline started.\n")

    for step_index, step in enumerate(PIPELINE_STEPS, start=1):
        print("=" * 80)
        print(f"[{step_index}/{len(PIPELINE_STEPS)}] {step.name}")
        print(f"Module: {step.module}")
        print("=" * 80)

        run_module(step.module)

        print(f"Completed: {step.name}\n")

    print("=" * 80)
    print("Running test suite")
    print("=" * 80)

    run_tests()

    print("\nFull automata pipeline completed successfully.")
    print("Generated outputs:")
    print("- reports/results/*.json")
    print("- reports/results/*.csv")
    print("- reports/figures/*.png")
    print("- reports/tables/*.csv")


if __name__ == "__main__":
    main()