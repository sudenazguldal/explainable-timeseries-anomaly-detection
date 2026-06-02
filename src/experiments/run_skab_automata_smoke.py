import json
from pathlib import Path

from src.config.load_config import load_config
from src.data.skab_loader import load_skab_dataset, get_skab_feature_columns
from src.data.label_utils import normalize_skab_labels
from src.data.splitter import create_skab_group_folds, validate_no_group_leakage
from src.preprocessing.scaler import fit_transform_train, transform_with_fitted_scaler
from src.preprocessing.pca import fit_transform_train_pca, transform_with_fitted_pca
from src.models.automata.paa import paa_transform
from src.models.automata.sax import sax_transform
from src.models.automata.probabilistic_automata import (
    ProbabilisticAutomata,
    extract_sliding_windows,
)
from src.models.automata.explainability import (
    explain_single_transition,
    explain_pattern_sequence,
)


def series_to_symbol_sequence(
    values,
    n_segments: int,
    alphabet_size: int,
) -> str:
    """
    Converts a one-dimensional numeric series into a symbolic SAX sequence.

    Pipeline:
    numeric series -> PAA -> SAX
    """
    paa_values = paa_transform(values, n_segments=n_segments)
    symbol_sequence = sax_transform(paa_values, alphabet_size=alphabet_size)

    return symbol_sequence


def run_skab_automata_smoke(config: dict) -> dict:
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

    train_indices, test_indices = folds[0]

    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    no_group_leakage = validate_no_group_leakage(
        train_df=train_df,
        test_df=test_df,
        group_column=skab_config["group_column"],
    )

    if not no_group_leakage:
        raise ValueError("Group leakage detected in SKAB smoke split.")

    scaled_train_df, scaler = fit_transform_train(
        train_df=train_df,
        feature_columns=feature_columns,
        method=preprocessing_config["normalization"],
    )

    scaled_test_df = transform_with_fitted_scaler(
        df=test_df,
        feature_columns=feature_columns,
        scaler=scaler,
    )

    pca_train_df, pca = fit_transform_train_pca(
        train_df=scaled_train_df,
        feature_columns=feature_columns,
        n_components=preprocessing_config["pca_components"],
        output_column="pc1",
    )

    pca_test_df = transform_with_fitted_pca(
        df=scaled_test_df,
        feature_columns=feature_columns,
        pca=pca,
        output_column="pc1",
    )

    window_size = automata_config["fixed"]["window_size"]
    alphabet_size = automata_config["fixed"]["alphabet_size"]

    configured_paa_segments = automata_config.get("paa_segments", 256)

    train_paa_segments = min(configured_paa_segments, len(pca_train_df))
    test_paa_segments = min(configured_paa_segments, len(pca_test_df))

    if train_paa_segments <= window_size:
        raise ValueError(
            "PAA segment count must be greater than automata window size."
        )

    if test_paa_segments <= window_size:
        raise ValueError(
            "Test PAA segment count must be greater than automata window size."
        )

    train_symbol_sequence = series_to_symbol_sequence(
        values=pca_train_df["pc1"].to_numpy(),
        n_segments=train_paa_segments,
        alphabet_size=alphabet_size,
    )

    test_symbol_sequence = series_to_symbol_sequence(
        values=pca_test_df["pc1"].to_numpy(),
        n_segments=test_paa_segments,
        alphabet_size=alphabet_size,
    )

    automata = ProbabilisticAutomata(
        window_size=window_size,
        smoothing=automata_config["smoothing"],
    )

    automata.fit(train_symbol_sequence)

    train_patterns = extract_sliding_windows(
        symbol_sequence=train_symbol_sequence,
        window_size=window_size,
    )

    test_patterns = extract_sliding_windows(
        symbol_sequence=test_symbol_sequence,
        window_size=window_size,
    )

    unseen_test_patterns = [
        pattern for pattern in test_patterns
        if pattern not in automata.states
    ]

    previous_state = sorted(automata.states)[0]
    incoming_pattern = (
        unseen_test_patterns[0]
        if unseen_test_patterns
        else test_patterns[0]
    )

    sample_transition_explanation = explain_single_transition(
        automata=automata,
        previous_state=previous_state,
        incoming_pattern=incoming_pattern,
        time_step=1,
        anomaly_threshold=0.05,
    )

    sample_path_explanation = explain_pattern_sequence(
        automata=automata,
        patterns=train_patterns[: min(5, len(train_patterns))],
        start_time_step=1,
        anomaly_threshold=0.05,
    )

    report = {
        "dataset": "SKAB",
        "target_column": skab_config["target_column"],
        "normalized_target_column": "label",
        "group_column": skab_config["group_column"],
        "feature_column_count": len(feature_columns),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_group_count": train_df[skab_config["group_column"]].nunique(),
        "test_group_count": test_df[skab_config["group_column"]].nunique(),
        "no_group_leakage": no_group_leakage,
        "paa_segments": {
            "configured": configured_paa_segments,
            "train_used": train_paa_segments,
            "test_used": test_paa_segments,
        },
        "automata_parameters": {
            "window_size": window_size,
            "alphabet_size": alphabet_size,
            "smoothing": automata_config["smoothing"],
        },
        "automata_summary": {
            "state_count": automata.state_count(),
            "transition_density": automata.transition_density(),
            "train_pattern_count": len(train_patterns),
            "test_pattern_count": len(test_patterns),
            "unseen_test_pattern_count": len(unseen_test_patterns),
        },
        "sample_transition_explanation": sample_transition_explanation,
        "sample_path_explanation": sample_path_explanation,
    }

    return report


def main() -> None:
    config = load_config("config.yaml")

    report = run_skab_automata_smoke(config)

    output_path = Path("reports/results/automata_smoke_skab.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("SKAB automata smoke run completed.")
    print(f"Report written to: {output_path}")

    print("\nAutomata summary")
    print(f"State count: {report['automata_summary']['state_count']}")
    print(f"Transition density: {report['automata_summary']['transition_density']}")
    print(f"Train pattern count: {report['automata_summary']['train_pattern_count']}")
    print(f"Test pattern count: {report['automata_summary']['test_pattern_count']}")
    print(f"Unseen test pattern count: {report['automata_summary']['unseen_test_pattern_count']}")
    print(f"No group leakage: {report['no_group_leakage']}")

    print("\nSample explanation")
    print(json.dumps(report["sample_transition_explanation"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()