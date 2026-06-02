import json
from pathlib import Path

from src.config.load_config import load_config
from src.data.batadal_loader import (
    load_batadal_dataset,
    detect_batadal_target_column,
    detect_batadal_time_column,
    get_batadal_feature_columns,
    split_batadal_time_ordered,
)
from src.data.label_utils import normalize_batadal_labels
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


def evaluate_unseen_patterns_only(
    automata: ProbabilisticAutomata,
    test_patterns: list[str],
    test_pattern_labels: list[int],
    anomaly_threshold: float,
) -> dict:
    """
    Evaluates only unseen incoming test patterns.

    Unseen pattern definition:
    - A test pattern is unseen if it does not exist in the training automata states.
    """
    if len(test_patterns) != len(test_pattern_labels):
        raise ValueError("test_patterns and test_pattern_labels must have the same length.")

    if len(test_patterns) < 2:
        raise ValueError("At least two test patterns are required for unseen evaluation.")

    current_state, _, _ = automata.resolve_pattern(test_patterns[0])

    y_true = []
    y_pred = []
    probabilities = []
    edit_distances = []
    explanations = []
    path_probability_so_far = 1.0

    total_transition_count = 0
    unseen_transition_count = 0

    for index, incoming_pattern in enumerate(test_patterns[1:], start=1):
        total_transition_count += 1

        is_unseen = incoming_pattern not in automata.states

        transition_result = automata.evaluate_transition(
            previous_state=current_state,
            incoming_pattern=incoming_pattern,
        )
        path_probability_so_far *= float(transition_result.probability)

        if is_unseen:
            unseen_transition_count += 1

            prediction = probability_to_binary_prediction(
                probability=transition_result.probability,
                anomaly_threshold=anomaly_threshold,
            )

            y_true.append(int(test_pattern_labels[index]))
            y_pred.append(prediction)
            probabilities.append(float(transition_result.probability))

            if transition_result.edit_distance is not None:
                edit_distances.append(int(transition_result.edit_distance))

            decision = "anomaly" if prediction == 1 else "normal"
            decision_reason = (
                "low_probability_path"
                if prediction == 1
                else "high_probability_path"
            )

            explanations.append({
                "time_step": index,
                "previous_state": transition_result.previous_state,
                "pattern": transition_result.incoming_pattern,
                "status": transition_result.status,
                "mapped_to": transition_result.resolved_state,
                "probability": float(transition_result.probability),
                "prediction": prediction,
                "true_label": int(test_pattern_labels[index]),
                "edit_distance": transition_result.edit_distance,
                "path_probability_so_far": path_probability_so_far,
                "decision": decision,
                "confidence": path_probability_so_far,
                "decision_reason": decision_reason,

            })

        current_state = transition_result.resolved_state

    if y_true:
        metrics = calculate_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
        )

        confusion = calculate_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
        )
    else:
        metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }
        confusion = [[0, 0], [0, 0]]

    unseen_ratio = (
        unseen_transition_count / total_transition_count
        if total_transition_count > 0
        else 0.0
    )

    average_edit_distance = (
        sum(edit_distances) / len(edit_distances)
        if edit_distances
        else 0.0
    )

    return {
        "metrics": metrics,
        "confusion_matrix": confusion,
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probabilities,
        "unseen_transition_count": unseen_transition_count,
        "total_transition_count": total_transition_count,
        "unseen_ratio": unseen_ratio,
        "average_edit_distance": average_edit_distance,
        "sample_explanations": explanations[:10],
    }


def run_batadal_automata_unseen_metrics(config: dict) -> dict:
    batadal_config = config["datasets"]["batadal"]
    preprocessing_config = config["preprocessing"]
    automata_config = config["automata"]

    anomaly_threshold = automata_config["anomaly_threshold"]

    df = load_batadal_dataset(
        raw_path=batadal_config["raw_path"],
    )

    target_column = detect_batadal_target_column(
        df=df,
        target_column_candidates=batadal_config["target_column_candidates"],
    )

    time_column = detect_batadal_time_column(
        df=df,
        time_column_candidates=batadal_config["time_column_candidates"],
    )

    df = normalize_batadal_labels(
        df=df,
        target_column=target_column,
        output_column="label",
    )

    feature_columns = get_batadal_feature_columns(
        df=df,
        target_column=target_column,
        time_column=time_column,
    )

    split_config = batadal_config["split"]

    train_df, validation_df, test_df = split_batadal_time_ordered(
        df=df,
        train_ratio=split_config["train"],
        validation_ratio=split_config["validation"],
        test_ratio=split_config["test"],
        time_column=time_column,
    )

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
        fallback_probability=automata_config["fallback_probability"],
    )

    automata.fit(train_symbol_sequence)

    evaluation = evaluate_unseen_patterns_only(
        automata=automata,
        test_patterns=test_patterns,
        test_pattern_labels=test_pattern_labels,
        anomaly_threshold=anomaly_threshold,
    )

    return {
        "dataset": "BATADAL",
        "scenario": "unseen",
        "target_column": target_column,
        "time_column": time_column,
        "feature_column_count": len(feature_columns),
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "automata_parameters": {
            "window_size": window_size,
            "alphabet_size": alphabet_size,
            "paa_segments": configured_paa_segments,
            "fallback_probability": automata_config["fallback_probability"],
            "anomaly_threshold": anomaly_threshold,
        },
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


def main() -> None:
    config = load_config("config.yaml")

    report = run_batadal_automata_unseen_metrics(config)

    output_path = Path("reports/results/automata_unseen_metrics_batadal.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("BATADAL automata unseen metric evaluation completed.")
    print(f"Report written to: {output_path}")

    print("\nMetrics")
    for metric_name, metric_value in report["metrics"].items():
        print(f"{metric_name}: {metric_value}")

    print("\nAutomata summary")
    for key, value in report["automata_summary"].items():
        print(f"{key}: {value}")

    print("\nSample unseen explanation")
    if report["sample_explanations"]:
        print(json.dumps(report["sample_explanations"][0], indent=2, ensure_ascii=False))
    else:
        print("No unseen pattern found.")


if __name__ == "__main__":
    main()