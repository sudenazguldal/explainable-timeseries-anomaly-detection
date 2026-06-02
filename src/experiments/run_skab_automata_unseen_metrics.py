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
from src.models.automata.pattern_labels import (
    aggregate_labels_by_segments,
    create_pattern_labels,
)
from src.experiments.run_batadal_automata_unseen_metrics import (
    evaluate_unseen_patterns_only,
)


def series_to_symbol_sequence(values, n_segments: int, alphabet_size: int) -> str:
    paa_values = paa_transform(values, n_segments=n_segments)
    return sax_transform(paa_values, alphabet_size=alphabet_size)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0

    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)

    return variance ** 0.5


def summarize_fold_metrics(fold_reports: list[dict]) -> dict:
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    summary = {}

    for metric_name in metric_names:
        values = [
            fold_report["metrics"][metric_name]
            for fold_report in fold_reports
        ]

        summary[metric_name] = {
            "mean": mean(values),
            "std": sample_std(values),
        }

    return summary


def summarize_unseen_statistics(fold_reports: list[dict]) -> dict:
    unseen_counts = [
        fold_report["automata_summary"]["unseen_test_pattern_count"]
        for fold_report in fold_reports
    ]

    unseen_ratios = [
        fold_report["automata_summary"]["unseen_ratio"]
        for fold_report in fold_reports
    ]

    edit_distances = [
        fold_report["automata_summary"]["average_edit_distance"]
        for fold_report in fold_reports
    ]

    return {
        "unseen_test_pattern_count_mean": mean(unseen_counts),
        "unseen_test_pattern_count_std": sample_std(unseen_counts),
        "unseen_ratio_mean": mean(unseen_ratios),
        "unseen_ratio_std": sample_std(unseen_ratios),
        "average_edit_distance_mean": mean(edit_distances),
        "average_edit_distance_std": sample_std(edit_distances),
    }


def run_single_skab_unseen_fold(
    df,
    train_indices: list[int],
    test_indices: list[int],
    feature_columns: list[str],
    skab_config: dict,
    preprocessing_config: dict,
    automata_config: dict,
    fold_index: int,
) -> dict:
    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    no_group_leakage = validate_no_group_leakage(
        train_df=train_df,
        test_df=test_df,
        group_column=skab_config["group_column"],
    )

    if not no_group_leakage:
        raise ValueError(f"Group leakage detected in SKAB unseen fold {fold_index}.")

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
    anomaly_threshold = automata_config["anomaly_threshold"]
    configured_paa_segments = automata_config.get("paa_segments", 256)

    train_paa_segments = min(configured_paa_segments, len(pca_train_df))
    test_paa_segments = min(configured_paa_segments, len(pca_test_df))

    if train_paa_segments <= window_size:
        raise ValueError("Train PAA segment count must be greater than window size.")

    if test_paa_segments <= window_size:
        raise ValueError("Test PAA segment count must be greater than window size.")

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

    test_segment_labels = aggregate_labels_by_segments(
        labels=pca_test_df["label"].to_numpy(),
        n_segments=test_paa_segments,
        strategy="any",
    )

    train_patterns = extract_sliding_windows(
        symbol_sequence=train_symbol_sequence,
        window_size=window_size,
    )

    test_patterns = extract_sliding_windows(
        symbol_sequence=test_symbol_sequence,
        window_size=window_size,
    )

    test_pattern_labels = create_pattern_labels(
        segment_labels=test_segment_labels,
        window_size=window_size,
        strategy="any",
    )

    automata = ProbabilisticAutomata(
        window_size=window_size,
        smoothing=automata_config["smoothing"],
    )

    automata.fit(train_symbol_sequence)

    evaluation = evaluate_unseen_patterns_only(
        automata=automata,
        test_patterns=test_patterns,
        test_pattern_labels=test_pattern_labels,
        anomaly_threshold=anomaly_threshold,
    )

    return {
        "fold_index": fold_index,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_group_count": train_df[skab_config["group_column"]].nunique(),
        "test_group_count": test_df[skab_config["group_column"]].nunique(),
        "no_group_leakage": no_group_leakage,
        "automata_summary": {
            "state_count": automata.state_count(),
            "transition_density": automata.transition_density(),
            "train_pattern_count": len(train_patterns),
            "test_pattern_count": len(test_patterns),
            "unseen_test_pattern_count": evaluation["unseen_transition_count"],
            "unseen_ratio": evaluation["unseen_ratio"],
            "average_edit_distance": evaluation["average_edit_distance"],
        },
        "metrics": evaluation["metrics"],
        "confusion_matrix": evaluation["confusion_matrix"],
        "sample_explanations": evaluation["sample_explanations"],
    }


def run_skab_automata_unseen_metrics(config: dict) -> dict:
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

    fold_reports = []

    for fold_index, (train_indices, test_indices) in enumerate(folds, start=1):
        fold_report = run_single_skab_unseen_fold(
            df=df,
            train_indices=train_indices,
            test_indices=test_indices,
            feature_columns=feature_columns,
            skab_config=skab_config,
            preprocessing_config=preprocessing_config,
            automata_config=automata_config,
            fold_index=fold_index,
        )

        fold_reports.append(fold_report)

    report = {
        "dataset": "SKAB",
        "scenario": "unseen",
        "target_column": skab_config["target_column"],
        "normalized_target_column": "label",
        "group_column": skab_config["group_column"],
        "feature_column_count": len(feature_columns),
        "fold_count": len(fold_reports),
        "automata_parameters": {
            "window_size": automata_config["fixed"]["window_size"],
            "alphabet_size": automata_config["fixed"]["alphabet_size"],
            "paa_segments": automata_config.get("paa_segments", 256),
            "smoothing": automata_config["smoothing"],
            "anomaly_threshold": automata_config["anomaly_threshold"],
        },
        "metric_summary": summarize_fold_metrics(fold_reports),
        "unseen_summary": summarize_unseen_statistics(fold_reports),
        "folds": fold_reports,
    }

    return report


def main() -> None:
    config = load_config("config.yaml")

    report = run_skab_automata_unseen_metrics(config)

    output_path = Path("reports/results/automata_unseen_metrics_skab.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("SKAB automata unseen metric evaluation completed.")
    print(f"Report written to: {output_path}")

    print("\nMetric summary")
    for metric_name, values in report["metric_summary"].items():
        print(f"{metric_name}: mean={values['mean']}, std={values['std']}")

    print("\nUnseen summary")
    for key, value in report["unseen_summary"].items():
        print(f"{key}: {value}")

    print("\nFold summaries")
    for fold in report["folds"]:
        print(
            f"Fold {fold['fold_index']}: "
            f"f1={fold['metrics']['f1_score']}, "
            f"unseen_ratio={fold['automata_summary']['unseen_ratio']}, "
            f"average_edit_distance={fold['automata_summary']['average_edit_distance']}, "
            f"no_group_leakage={fold['no_group_leakage']}"
        )

    print("\nSample unseen explanation")
    first_fold_with_example = next(
        (
            fold for fold in report["folds"]
            if fold["sample_explanations"]
        ),
        None,
    )

    if first_fold_with_example is not None:
        print(json.dumps(
            first_fold_with_example["sample_explanations"][0],
            indent=2,
            ensure_ascii=False,
        ))
    else:
        print("No unseen pattern found.")


if __name__ == "__main__":
    main()