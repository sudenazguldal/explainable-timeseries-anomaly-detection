from collections import Counter, defaultdict
from dataclasses import dataclass, field

from src.models.automata.levenshtein import find_nearest_pattern


def extract_sliding_windows(symbol_sequence: str, window_size: int) -> list[str]:
    """
    Extracts fixed-size patterns from a symbolic sequence using sliding window.

    Example:
    sequence = "abcde", window_size = 3
    patterns = ["abc", "bcd", "cde"]
    """
    if window_size <= 0:
        raise ValueError("Window size must be positive.")

    if window_size > len(symbol_sequence):
        raise ValueError("Window size cannot exceed symbolic sequence length.")

    return [
        symbol_sequence[index:index + window_size]
        for index in range(len(symbol_sequence) - window_size + 1)
    ]


def build_transition_counts(patterns: list[str]) -> dict[str, Counter]:
    """
    Builds transition counts between consecutive patterns. Each unique pattern is treated as a state.
    """
    transition_counts: dict[str, Counter] = defaultdict(Counter)

    for current_state, next_state in zip(patterns, patterns[1:]):
        transition_counts[current_state][next_state] += 1

    return dict(transition_counts)


def build_transition_probabilities(
    transition_counts: dict[str, Counter],
) -> dict[str, dict[str, float]]:
    
    """
    Converts transition counts into frequency-based transition probabilities.

    P(S_i -> S_j) = count(S_i -> S_j) / total_outgoing(S_i)
    """

    transition_probabilities: dict[str, dict[str, float]] = {}

    for current_state, next_state_counts in transition_counts.items():
        total_outgoing = sum(next_state_counts.values())

        if total_outgoing == 0:
            transition_probabilities[current_state] = {}
            continue

        transition_probabilities[current_state] = {
            next_state: count / total_outgoing
            for next_state, count in next_state_counts.items()
        }

    return transition_probabilities


def calculate_transition_density(states: set[str], transition_counts: dict[str, Counter]) -> float:
    """
    Calculates transition density. Density = observed directed transitions / possible directed transitions
    """
    if not states:
        return 0.0

    observed_transition_count = sum(
        len(next_states)
        for next_states in transition_counts.values()
    )

    possible_transition_count = len(states) * len(states)

    return observed_transition_count / possible_transition_count


@dataclass
class TransitionResult:
    previous_state: str
    incoming_pattern: str
    resolved_state: str
    status: str
    probability: float
    edit_distance: int | None = None


@dataclass
class ProbabilisticAutomata:
    """
    Frequency-based probabilistic automata for symbolic time-series patterns.
    """
    window_size: int
    smoothing: float = 1e-6
    states: set[str] = field(default_factory=set)
    transition_counts: dict[str, Counter] = field(default_factory=dict)
    transition_probabilities: dict[str, dict[str, float]] = field(default_factory=dict)

    def fit(self, symbol_sequence: str) -> "ProbabilisticAutomata":
        patterns = extract_sliding_windows(
            symbol_sequence=symbol_sequence,
            window_size=self.window_size,
        )

        self.states = set(patterns)
        self.transition_counts = build_transition_counts(patterns)
        self.transition_probabilities = build_transition_probabilities(
            self.transition_counts
        )

        return self

    def state_count(self) -> int:
        return len(self.states)

    def transition_density(self) -> float:
        return calculate_transition_density(
            states=self.states,
            transition_counts=self.transition_counts,
        )

    def resolve_pattern(self, pattern: str) -> tuple[str, str, int | None]:
        """
        Resolves an incoming pattern.

        If the pattern was seen during training, it is used directly.
        If unseen, the nearest known pattern is selected with Levenshtein distance.
        """
        if pattern in self.states:
            return pattern, "seen", None

        nearest_pattern, distance = find_nearest_pattern(
            pattern=pattern,
            known_patterns=self.states,
        )

        return nearest_pattern, "unseen", distance

    def transition_probability(self, previous_state: str, next_state: str) -> float:
        """
        Returns transition probability. If the transition was not observed, returns smoothing probability.
        """
        return self.transition_probabilities.get(
            previous_state,
            {},
        ).get(
            next_state,
            self.smoothing,
        )

    def evaluate_transition(self, previous_state: str, incoming_pattern: str) -> TransitionResult:
        """
        Evaluates a single transition and handles unseen incoming patterns.
        """
        resolved_state, status, edit_distance = self.resolve_pattern(incoming_pattern)

        probability = self.transition_probability(
            previous_state=previous_state,
            next_state=resolved_state,
        )

        return TransitionResult(
            previous_state=previous_state,
            incoming_pattern=incoming_pattern,
            resolved_state=resolved_state,
            status=status,
            probability=probability,
            edit_distance=edit_distance,
        )

    def path_probability(self, patterns: list[str]) -> float:
        """
        Computes path probability as the product of consecutive transition probabilities.
        """
        if len(patterns) < 2:
            return 1.0

        probability = 1.0

        for previous_state, incoming_pattern in zip(patterns, patterns[1:]):
            result = self.evaluate_transition(
                previous_state=previous_state,
                incoming_pattern=incoming_pattern,
            )
            probability *= result.probability

        return probability