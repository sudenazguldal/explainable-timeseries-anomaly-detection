from src.models.automata.explainability import (
    decide_from_probability,
    explain_pattern_sequence,
    explain_single_transition,
)
from src.models.automata.probabilistic_automata import ProbabilisticAutomata


def test_decide_from_probability_marks_low_probability_as_anomaly():
    decision = decide_from_probability(
        probability=0.01,
        anomaly_threshold=0.05,
    )

    assert decision == "anomaly"


def test_decide_from_probability_marks_high_probability_as_normal():
    decision = decide_from_probability(
        probability=0.80,
        anomaly_threshold=0.05,
    )

    assert decision == "normal"


def test_explain_single_transition_for_seen_pattern():
    automata = ProbabilisticAutomata(window_size=3)
    automata.fit("aaabbc")

    explanation = explain_single_transition(
        automata=automata,
        previous_state="aaa",
        incoming_pattern="aab",
        time_step=5,
        anomaly_threshold=0.05,
    )

    assert explanation["time_step"] == 5
    assert explanation["state"] == "aaa"
    assert explanation["pattern"] == "aab"
    assert explanation["status"] == "seen"
    assert explanation["mapped_to"] == "aab"
    assert explanation["probability"] == 1.0
    assert explanation["decision"] == "normal"
    assert explanation["confidence"] == 1.0


def test_explain_single_transition_for_unseen_pattern():
    automata = ProbabilisticAutomata(window_size=3)
    automata.fit("aaabbc")

    explanation = explain_single_transition(
        automata=automata,
        previous_state="aaa",
        incoming_pattern="aac",
        time_step=6,
        anomaly_threshold=0.05,
    )

    assert explanation["time_step"] == 6
    assert explanation["state"] == "aaa"
    assert explanation["pattern"] == "aac"
    assert explanation["status"] == "unseen"
    assert explanation["mapped_to"] in automata.states
    assert explanation["edit_distance"] == 1
    assert "probability" in explanation
    assert "decision" in explanation
    assert "confidence" in explanation


def test_explain_pattern_sequence_returns_path_probability_and_transitions():
    automata = ProbabilisticAutomata(window_size=3)
    automata.fit("aaabbc")

    explanation = explain_pattern_sequence(
        automata=automata,
        patterns=["aaa", "aab", "abb"],
        start_time_step=10,
        anomaly_threshold=0.05,
    )

    assert explanation["start_time_step"] == 10
    assert explanation["end_time_step"] == 12
    assert explanation["path_probability"] == 1.0
    assert explanation["decision"] == "normal"
    assert explanation["confidence"] == 1.0
    assert len(explanation["transitions"]) == 2
    assert explanation["transitions"][0]["state"] == "aaa"
    assert explanation["transitions"][0]["pattern"] == "aab"


def test_explain_pattern_sequence_handles_single_pattern():
    automata = ProbabilisticAutomata(window_size=3)
    automata.fit("aaabbc")

    explanation = explain_pattern_sequence(
        automata=automata,
        patterns=["aaa"],
        start_time_step=3,
        anomaly_threshold=0.05,
    )

    assert explanation["start_time_step"] == 3
    assert explanation["end_time_step"] == 3
    assert explanation["transitions"] == []
    assert explanation["path_probability"] == 1.0
    assert explanation["decision"] == "normal"