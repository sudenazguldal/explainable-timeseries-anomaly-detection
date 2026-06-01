from src.models.automata.probabilistic_automata import (
    ProbabilisticAutomata,
    TransitionResult,
)


def decide_from_probability(
    probability: float,
    anomaly_threshold: float,
) -> str:
    """
    Converts a probability score into a decision.

    Lower transition/path probability indicates more unexpected behavior.
    """
    if probability < anomaly_threshold:
        return "anomaly"

    return "normal"


def transition_result_to_explanation(
    time_step: int,
    transition_result: TransitionResult,
    anomaly_threshold: float,
) -> dict:
    """
    Converts a TransitionResult into the required JSON-compatible explanation.

    Required fields include:
    - current/previous state
    - observed incoming pattern
    - seen/unseen status
    - mapped state
    - transition probability
    - final decision
    - confidence score
    """
    decision = decide_from_probability(
        probability=transition_result.probability,
        anomaly_threshold=anomaly_threshold,
    )

    explanation = {
        "time_step": time_step,
        "state": transition_result.previous_state,
        "pattern": transition_result.incoming_pattern,
        "status": transition_result.status,
        "mapped_to": transition_result.resolved_state,
        "probability": transition_result.probability,
        "decision": decision,
        "confidence": transition_result.probability,
    }

    if transition_result.edit_distance is not None:
        explanation["edit_distance"] = transition_result.edit_distance

    return explanation


def explain_single_transition(
    automata: ProbabilisticAutomata,
    previous_state: str,
    incoming_pattern: str,
    time_step: int,
    anomaly_threshold: float,
) -> dict:
    """
    Produces a JSON-compatible explanation for a single automata transition.
    """
    transition_result = automata.evaluate_transition(
        previous_state=previous_state,
        incoming_pattern=incoming_pattern,
    )

    return transition_result_to_explanation(
        time_step=time_step,
        transition_result=transition_result,
        anomaly_threshold=anomaly_threshold,
    )


def explain_pattern_sequence(
    automata: ProbabilisticAutomata,
    patterns: list[str],
    start_time_step: int,
    anomaly_threshold: float,
) -> dict:
    """
    Produces a JSON-compatible explanation for a sequence of patterns.

    Path probability is calculated as the product of consecutive transition
    probabilities.
    """
    if len(patterns) < 2:
        return {
            "start_time_step": start_time_step,
            "end_time_step": start_time_step,
            "transitions": [],
            "path_probability": 1.0,
            "decision": "normal",
            "confidence": 1.0,
        }

    transition_explanations = []
    path_probability = 1.0

    for offset, (previous_state, incoming_pattern) in enumerate(
        zip(patterns, patterns[1:]),
        start=1,
    ):
        time_step = start_time_step + offset

        transition_result = automata.evaluate_transition(
            previous_state=previous_state,
            incoming_pattern=incoming_pattern,
        )

        path_probability *= transition_result.probability

        transition_explanation = transition_result_to_explanation(
            time_step=time_step,
            transition_result=transition_result,
            anomaly_threshold=anomaly_threshold,
        )

        transition_explanations.append(transition_explanation)

    sequence_decision = decide_from_probability(
        probability=path_probability,
        anomaly_threshold=anomaly_threshold,
    )

    return {
        "start_time_step": start_time_step,
        "end_time_step": start_time_step + len(patterns) - 1,
        "transitions": transition_explanations,
        "path_probability": path_probability,
        "decision": sequence_decision,
        "confidence": path_probability,
    }