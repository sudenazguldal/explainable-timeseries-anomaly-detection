from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar


def run_wilcoxon_test(
    model_a_scores: list[float],
    model_b_scores: list[float],
    alternative: str = "two-sided",
) -> dict:
    """
    Runs Wilcoxon signed-rank test for paired metric scores.

    Intended use:  Compare fold-level F1 scores of Automata vs Deep Learning.
    """
    if len(model_a_scores) != len(model_b_scores):
        raise ValueError("Score lists must have the same length.")

    if len(model_a_scores) == 0:
        raise ValueError("Score lists cannot be empty.")

    statistic, p_value = wilcoxon(
        model_a_scores,
        model_b_scores,
        alternative=alternative,
    )

    return {
        "test": "wilcoxon_signed_rank",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alternative": alternative,
        "n_pairs": len(model_a_scores),
    }


def build_mcnemar_table(
    y_true: list[int],
    model_a_pred: list[int],
    model_b_pred: list[int],
) -> list[list[int]]:
    """
    Builds McNemar contingency table.

    Table format:
    [    [both_correct, model_a_correct_model_b_wrong],
         [model_a_wrong_model_b_correct, both_wrong]      ]
    """
    if not (len(y_true) == len(model_a_pred) == len(model_b_pred)):
        raise ValueError("y_true and prediction lists must have the same length.")

    both_correct = 0
    a_correct_b_wrong = 0
    a_wrong_b_correct = 0
    both_wrong = 0

    for true_label, pred_a, pred_b in zip(y_true, model_a_pred, model_b_pred):
        a_correct = pred_a == true_label
        b_correct = pred_b == true_label

        if a_correct and b_correct:
            both_correct += 1
        elif a_correct and not b_correct:
            a_correct_b_wrong += 1
        elif not a_correct and b_correct:
            a_wrong_b_correct += 1
        else:
            both_wrong += 1

    return [
        [both_correct, a_correct_b_wrong],
        [a_wrong_b_correct, both_wrong],
    ]


def run_mcnemar_test(
    y_true: list[int],
    model_a_pred: list[int],
    model_b_pred: list[int],
    exact: bool = True,
) -> dict:
    """
    Runs McNemar test for paired classifier predictions.

    Intended use: Compare Automata vs Deep Learning predictions on the same test samples.
    """
    table = build_mcnemar_table(
        y_true=y_true,
        model_a_pred=model_a_pred,
        model_b_pred=model_b_pred,
    )

    result = mcnemar(
        table,
        exact=exact,
    )

    return {
        "test": "mcnemar",
        "table": table,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "exact": exact,
    }