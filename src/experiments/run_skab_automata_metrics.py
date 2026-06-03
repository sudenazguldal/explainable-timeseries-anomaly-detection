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
from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_confusion_matrix,
    probability_to_binary_prediction,
)


def series_to_symbol_sequence(values, n_segments: int, alphabet_size: int) -> str:
    paa_values = paa_transform(values, n_segments=n_segments)
    return sax_transform(paa_values, alphabet_size=alphabet_size)


def evaluate_automata_on_patterns(
    automata: ProbabilisticAutomata,
    test_patterns: list[str],
    test_pattern_labels: list[int],
    anomaly_threshold: float,
) -> dict:
    if len(test_patterns) != len(test_pattern_labels):
        raise ValueError("test_patterns and test_pattern_labels must have the same length.")

    if len(test_patterns) < 2:
        raise ValueError("At least two test patterns are required for transition evaluation.")

    current_state, _, _ = automata.resolve_pattern(test_patterns[0])

    y_true = []
    y_pred = []
    probabilities = []
    explanations = []
    path_probability_so_far = 1.0


    for index, incoming_pattern in enumerate(test_patterns[1:], start=1):
        transition_result = automata.evaluate_transition(
            previous_state=current_state,
            incoming_pattern=incoming_pattern,
        )

        prediction = probability_to_binary_prediction(
            probability=transition_result.probability,
            anomaly_threshold=anomaly_threshold,
        )

        path_probability_so_far *= float(transition_result.probability)

        decision = "anomaly" if prediction == 1 else "normal"
        decision_reason = (
            "low_probability_path"
            if prediction == 1
            else "high_probability_path"
        )
        y_true.append(int(test_pattern_labels[index]))
        y_pred.append(prediction)
        probabilities.append(float(transition_result.probability))

        explanations.append({
            "time_step": index,
            "previous_state": transition_result.previous_state,
            "pattern": transition_result.incoming_pattern,
            "status": transition_result.status,
            "mapped_to": transition_result.resolved_state,
            "probability": float(transition_result.probability),
            "path_probability_so_far": path_probability_so_far,
            "prediction": prediction,
            "decision": decision,
            "confidence": path_probability_so_far,
            "decision_reason": decision_reason,
            "true_label": int(test_pattern_labels[index]),
            "edit_distance": transition_result.edit_distance,
        })

        current_state = transition_result.resolved_state

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
    )

    confusion = calculate_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
    )

    return {
        "metrics": metrics,
        "confusion_matrix": confusion,
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probabilities,
        "sample_explanations": explanations[:10],
    }


def run_single_skab_fold(
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
        raise ValueError(f"Group leakage detected in SKAB fold {fold_index}.")

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

    train_segment_labels = aggregate_labels_by_segments(
        labels=pca_train_df["label"].to_numpy(),
        n_segments=train_paa_segments,
        strategy="any",
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

    train_pattern_labels = create_pattern_labels(
        segment_labels=train_segment_labels,
        window_size=window_size,
        strategy="any",
    )

    test_pattern_labels = create_pattern_labels(
        segment_labels=test_segment_labels,
        window_size=window_size,
        strategy="any",
    )

    automata = ProbabilisticAutomata(
        window_size=window_size,
        fallback_probability=automata_config["fallback_probability"],
    )

    automata.fit(train_symbol_sequence)

    evaluation = evaluate_automata_on_patterns(
        automata=automata,
        test_patterns=test_patterns,
        test_pattern_labels=test_pattern_labels,
        anomaly_threshold=anomaly_threshold,
    )

    unseen_test_patterns = [
        pattern for pattern in test_patterns
        if pattern not in automata.states
    ]

    unseen_test_pattern_count = len(unseen_test_patterns)
    test_pattern_count = len(test_patterns)

    unseen_ratio = (
        unseen_test_pattern_count / test_pattern_count
        if test_pattern_count > 0
        else 0.0
)

    return {
        "fold_index": fold_index,
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
        "automata_summary": {
            "state_count": automata.state_count(),
            "transition_density": automata.transition_density(),
            "train_pattern_count": len(train_patterns),
            "test_pattern_count": test_pattern_count,
            "unseen_test_pattern_count": unseen_test_pattern_count,
            "unseen_ratio": unseen_ratio,
            "train_anomaly_pattern_count": int(sum(train_pattern_labels)),
            "test_anomaly_pattern_count": int(sum(test_pattern_labels)),
        },
        "metrics": evaluation["metrics"],
        "confusion_matrix": evaluation["confusion_matrix"],
        "prediction_outputs": {
            "y_true": evaluation["y_true"],
            "y_pred": evaluation["y_pred"],
            "probabilities": evaluation["probabilities"],
            "anomaly_scores": [
                1.0 - probability
                for probability in evaluation["probabilities"]
            ],
        },
        "sample_explanations": evaluation["sample_explanations"],
    }


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


def run_skab_automata_metrics(config: dict) -> dict:
    skab_config = config["datasets"]["skab"]
    preprocessing_config = config["preprocessing"]
    automata_config = config["automata"]
    anomaly_threshold = automata_config["anomaly_threshold"]

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
        fold_report = run_single_skab_fold(
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
        "scenario": "original",
        "evaluation_scope": "full_test_transitions_with_unseen_handling",
        "target_column": skab_config["target_column"],
        "normalized_target_column": "label",
        "group_column": skab_config["group_column"],
        "feature_column_count": len(feature_columns),
        "fold_count": len(fold_reports),
        "automata_parameters": {
            "window_size": automata_config["fixed"]["window_size"],
            "alphabet_size": automata_config["fixed"]["alphabet_size"],
            "fallback_probability": automata_config["fallback_probability"],
            "anomaly_threshold": anomaly_threshold,
        },
        "metric_summary": summarize_fold_metrics(fold_reports),
        "folds": fold_reports,
    }

    return report


def main() -> None:
    config = load_config("config.yaml")

    report = run_skab_automata_metrics(config)

    output_path = Path("reports/results/automata_metrics_skab.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("SKAB automata metric evaluation completed.")
    print(f"Report written to: {output_path}")

    print("\nMetric summary")
    for metric_name, values in report["metric_summary"].items():
        print(f"{metric_name}: mean={values['mean']}, std={values['std']}")

    print("\nFold summaries")
    for fold in report["folds"]:
        print(
            f"Fold {fold['fold_index']}: "
            f"accuracy={fold['metrics']['accuracy']}, "
            f"precision={fold['metrics']['precision']}, "
            f"recall={fold['metrics']['recall']}, "
            f"f1={fold['metrics']['f1_score']}, "
            f"no_group_leakage={fold['no_group_leakage']}"
        )


if __name__ == "__main__":
    main()