from src.models.automata.probabilistic_automata import (
    ProbabilisticAutomata,
    build_transition_counts,
    build_transition_probabilities,
    calculate_transition_density,
    extract_sliding_windows,
)


def test_extract_sliding_windows_creates_patterns():
    patterns = extract_sliding_windows(
        symbol_sequence="abcde",
        window_size=3,
    )

    assert patterns == ["abc", "bcd", "cde"]


def test_build_transition_counts_counts_consecutive_patterns():
    patterns = ["aaa", "aab", "abb", "aab"]

    transition_counts = build_transition_counts(patterns)

    assert transition_counts["aaa"]["aab"] == 1
    assert transition_counts["aab"]["abb"] == 1
    assert transition_counts["abb"]["aab"] == 1


def test_build_transition_probabilities_are_frequency_based():
    patterns = ["aaa", "aab", "aaa", "abb", "aaa"]
    transition_counts = build_transition_counts(patterns)

    probabilities = build_transition_probabilities(transition_counts)

    assert probabilities["aaa"]["aab"] == 0.5
    assert probabilities["aaa"]["abb"] == 0.5


def test_calculate_transition_density():
    states = {"aaa", "aab", "abb"}
    transition_counts = build_transition_counts(["aaa", "aab", "abb"])

    density = calculate_transition_density(
        states=states,
        transition_counts=transition_counts,
    )

    assert density == 2 / 9


def test_probabilistic_automata_fit_creates_states_and_probabilities():
    automata = ProbabilisticAutomata(window_size=3)
    automata.fit("aaabbc")

    assert automata.state_count() == 4
    assert "aaa" in automata.states
    assert "aab" in automata.states
    assert automata.transition_probability("aaa", "aab") == 1.0


def test_probabilistic_automata_resolves_seen_pattern():
    automata = ProbabilisticAutomata(window_size=3)
    automata.fit("aaabbc")

    resolved_state, status, edit_distance = automata.resolve_pattern("aab")

    assert resolved_state == "aab"
    assert status == "seen"
    assert edit_distance is None


def test_probabilistic_automata_resolves_unseen_pattern_with_levenshtein():
    automata = ProbabilisticAutomata(window_size=3)
    automata.fit("aaabbc")

    resolved_state, status, edit_distance = automata.resolve_pattern("aac")

    assert resolved_state in automata.states
    assert status == "unseen"
    assert edit_distance == 1


def test_evaluate_transition_returns_probability_and_status():
    automata = ProbabilisticAutomata(window_size=3)
    automata.fit("aaabbc")

    result = automata.evaluate_transition(
        previous_state="aaa",
        incoming_pattern="aab",
    )

    assert result.previous_state == "aaa"
    assert result.incoming_pattern == "aab"
    assert result.resolved_state == "aab"
    assert result.status == "seen"
    assert result.probability == 1.0


def test_path_probability_multiplies_transition_probabilities():
    automata = ProbabilisticAutomata(window_size=3)
    automata.fit("aaabbc")

    probability = automata.path_probability(["aaa", "aab", "abb"])

    assert probability == 1.0

def test_transition_probability_casts_string_fallback_probability_to_float():
    automata = ProbabilisticAutomata(
        window_size=3,
        fallback_probability="1e-6",
    )
    automata.fit("aaabbc")

    probability = automata.transition_probability(
        previous_state="aaa",
        next_state="ccc",
    )

    assert probability == 1e-6
    assert isinstance(probability, float)