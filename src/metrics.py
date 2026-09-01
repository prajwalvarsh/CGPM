"""Metrics.

The paper reports three families and you should never show one without the
others, because each alone is trivially gameable:

    quality   how good the answers are (always-retrieve wins if you stop here)
    cost      how often the memory store was actually searched
    gating    how good the confidence estimate itself is, independent of the
              downstream generator (AUROC, ECE, risk-coverage AUC)

`summarize` bundles all three into the dictionary that becomes one row of the
results table in `experiments/log_template.md`.
"""

from __future__ import annotations

import re
import string
from typing import Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------
# Answer quality
# --------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Standard SQuAD-style token F1."""
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)
    common: Dict[str, int] = {}
    for token in predicted_tokens:
        if token in reference_tokens:
            common[token] = min(predicted_tokens.count(token),
                                reference_tokens.count(token))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def containment(prediction: str, reference: str) -> float:
    """Loose credit: does the answer contain the reference span?

    Useful for small models that answer correctly but verbosely. Report it
    alongside F1, never instead of it.
    """
    return float(normalize_answer(reference) in normalize_answer(prediction))


# --------------------------------------------------------------------------
# Gating quality
# --------------------------------------------------------------------------

def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve, computed from ranks (ties handled)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # Average ranks within tied groups.
    sorted_scores = scores[order]
    start = 0
    for end in range(1, len(sorted_scores) + 1):
        if end == len(sorted_scores) or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def expected_calibration_error(probabilities: Sequence[float],
                               labels: Sequence[int], bins: int = 10) -> float:
    """ECE with equal-width bins. Lower is better; report the bin count."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(probabilities) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probabilities > lower) & (probabilities <= upper)
        if not mask.any():
            continue
        weight = mask.mean()
        error += weight * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(error)


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return float(np.mean((probabilities - labels) ** 2))


def risk_coverage(scores: Sequence[float], correct: Sequence[float]
                  ) -> Dict[str, np.ndarray]:
    """Risk as a function of coverage, sweeping an abstention threshold.

    Answer the most confident turns first; risk is the error rate among the
    turns you chose to answer. The area under this curve is the single number
    that says whether the confidence estimate is useful for abstention.
    """
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=float)
    order = np.argsort(-scores)
    ordered_correct = correct[order]
    cumulative = np.cumsum(ordered_correct)
    counts = np.arange(1, len(ordered_correct) + 1)
    return {
        "coverage": counts / len(ordered_correct),
        "risk": 1.0 - cumulative / counts,
    }


def risk_coverage_auc(scores: Sequence[float], correct: Sequence[float]) -> float:
    curve = risk_coverage(scores, correct)
    return float(np.trapezoid(curve["risk"], curve["coverage"])
                 if hasattr(np, "trapezoid")
                 else np.trapz(curve["risk"], curve["coverage"]))


# --------------------------------------------------------------------------
# The row that goes into the paper
# --------------------------------------------------------------------------

def summarize(predictions: Sequence[str],
              references: Sequence[str],
              routes: Sequence[str],
              needs_memory: Sequence[int],
              gate_scores: Optional[Sequence[float]] = None,
              latencies_ms: Optional[Sequence[float]] = None,
              prompt_tokens: Optional[Sequence[int]] = None) -> Dict[str, float]:
    """Produce one results row: quality, cost and gating quality together."""
    f1_scores = [token_f1(p, r) for p, r in zip(predictions, references)]
    em_scores = [exact_match(p, r) for p, r in zip(predictions, references)]
    contain = [containment(p, r) for p, r in zip(predictions, references)]

    routes = list(routes)
    needs = np.asarray(list(needs_memory), dtype=int)
    retrieved = np.array([r == "retrieve" for r in routes], dtype=int)

    row: Dict[str, float] = {
        "n": float(len(predictions)),
        "f1": 100.0 * float(np.mean(f1_scores)) if f1_scores else 0.0,
        "em": 100.0 * float(np.mean(em_scores)) if em_scores else 0.0,
        "containment": 100.0 * float(np.mean(contain)) if contain else 0.0,
        "retrieval_call_rate": 100.0 * float(retrieved.mean()) if len(retrieved) else 0.0,
        "clarify_rate": 100.0 * float(np.mean([r == "clarify" for r in routes]))
                        if routes else 0.0,
    }

    # Split quality by whether memory was genuinely needed. A gate that looks
    # good on average while collapsing on memory-dependent turns is the most
    # common failure mode in this project, so surface it by default.
    if len(needs) == len(f1_scores) and len(needs):
        memory_mask = needs == 1
        if memory_mask.any():
            row["f1_memory_needed"] = 100.0 * float(np.mean(np.asarray(f1_scores)[memory_mask]))
        if (~memory_mask).any():
            row["f1_general"] = 100.0 * float(np.mean(np.asarray(f1_scores)[~memory_mask]))
        row["unnecessary_retrieval_rate"] = 100.0 * float(
            np.mean(retrieved[~memory_mask])) if (~memory_mask).any() else 0.0
        row["missed_retrieval_rate"] = 100.0 * float(
            np.mean(1 - retrieved[memory_mask])) if memory_mask.any() else 0.0

    if gate_scores is not None and len(gate_scores) == len(needs) and len(needs):
        row["gate_auroc"] = float(auroc(gate_scores, needs))
        row["gate_ece"] = float(expected_calibration_error(gate_scores, needs))
        row["gate_brier"] = float(brier_score(gate_scores, needs))
        row["risk_coverage_auc"] = float(risk_coverage_auc(gate_scores, f1_scores))

    if latencies_ms:
        row["latency_ms_mean"] = float(np.mean(latencies_ms))
        row["latency_ms_p90"] = float(np.percentile(latencies_ms, 90))
    if prompt_tokens:
        row["prompt_tokens_mean"] = float(np.mean(prompt_tokens))

    return row


def format_row(row: Dict[str, float], keys: Optional[List[str]] = None) -> str:
    """Render one results row as a Markdown table line."""
    keys = keys or list(row)
    return "| " + " | ".join(
        f"{row[k]:.1f}" if isinstance(row.get(k), float) else str(row.get(k, ""))
        for k in keys) + " |"
