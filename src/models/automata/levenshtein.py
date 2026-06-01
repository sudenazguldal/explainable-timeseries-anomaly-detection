def levenshtein_distance(left: str, right: str) -> int:
    """
    Computes Levenshtein edit distance between two strings.
    """
    if left == right:
        return 0

    if len(left) == 0:
        return len(right)

    if len(right) == 0:
        return len(left)

    previous_row = list(range(len(right) + 1))

    for i, left_char in enumerate(left, start=1):
        current_row = [i]

        for j, right_char in enumerate(right, start=1):
            insertion_cost = current_row[j - 1] + 1
            deletion_cost = previous_row[j] + 1
            substitution_cost = previous_row[j - 1]

            if left_char != right_char:
                substitution_cost += 1

            current_row.append(
                min(insertion_cost, deletion_cost, substitution_cost)
            )

        previous_row = current_row

    return previous_row[-1]


def find_nearest_pattern(pattern: str, known_patterns: set[str]) -> tuple[str, int]:
    """
    Finds the nearest known pattern using Levenshtein distance.

    Returns:
    - nearest pattern
    - edit distance
    """
    if not known_patterns:
        raise ValueError("Known pattern set cannot be empty.")

    nearest_pattern = None
    nearest_distance = None

    for candidate in sorted(known_patterns):
        distance = levenshtein_distance(pattern, candidate)

        if nearest_distance is None or distance < nearest_distance:
            nearest_pattern = candidate
            nearest_distance = distance

    return nearest_pattern, nearest_distance