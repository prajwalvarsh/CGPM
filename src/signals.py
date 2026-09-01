"""Cheap pre-retrieval signals.

Everything in this module must be computable WITHOUT searching the memory
index. That constraint is the whole point of the project, so it is enforced
here rather than left to discipline: `build_features` is never given a
`MemoryStore.search` result, only the store's sketch.

Three families of signal are assembled:

    surface   6 features from the raw text of the turn
    sketch    8 features from the query vector against the memory sketch
    probe     5 features from a short forward pass of the small language model

Ablating these three families against each other is the core of the paper's
analysis section, so keep the group boundaries intact when you add features.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .memory import MemorySketch
from .slm import ProbeSignals

SURFACE_WIDTH = 6
SKETCH_WIDTH = 8

_FIRST_PERSON = re.compile(r"\b(i|me|my|mine|myself|we|our)\b", re.IGNORECASE)
_TEMPORAL = re.compile(
    r"\b(yesterday|last|earlier|before|ago|previously|when|remind|remember|"
    r"session|we discussed|told you)\b", re.IGNORECASE)
_INTERROGATIVE = re.compile(r"\b(what|which|who|when|where|why|how)\b", re.IGNORECASE)


FEATURE_NAMES: List[str] = [
    # surface
    "surface/log_num_tokens",
    "surface/first_person_rate",
    "surface/temporal_marker_rate",
    "surface/has_interrogative",
    "surface/has_proper_noun",
    "surface/punctuation_rate",
    # sketch
    "sketch/best_centroid_sim",
    "sketch/second_centroid_sim",
    "sketch/centroid_margin",
    "sketch/occupancy_weighted_sim",
    "sketch/mean_centroid_sim",
    "sketch/std_centroid_sim",
    "sketch/inside_radius",
    "sketch/log_store_size",
    # probe
    "probe/first_token_entropy",
    "probe/mean_token_entropy",
    "probe/max_token_logprob",
    "probe/top1_top2_margin",
    "probe/says_memory",
]

GROUPS = {
    "surface": list(range(0, SURFACE_WIDTH)),
    "sketch": list(range(SURFACE_WIDTH, SURFACE_WIDTH + SKETCH_WIDTH)),
    "probe": list(range(SURFACE_WIDTH + SKETCH_WIDTH, len(FEATURE_NAMES))),
}

FEATURE_WIDTH = len(FEATURE_NAMES)


def surface_features(question: str) -> np.ndarray:
    """Six features read straight off the text of the turn."""
    tokens = question.split()
    num_tokens = max(len(tokens), 1)
    return np.array([
        float(np.log1p(num_tokens)),
        len(_FIRST_PERSON.findall(question)) / num_tokens,
        len(_TEMPORAL.findall(question)) / num_tokens,
        1.0 if _INTERROGATIVE.search(question) else 0.0,
        # A capitalised token that is not sentence-initial hints at a person,
        # place or project the assistant may have stored a note about.
        1.0 if any(t[:1].isupper() for t in tokens[1:]) else 0.0,
        sum(character in "?!,.;:" for character in question) / max(len(question), 1),
    ], dtype=np.float32)


@dataclass
class FeatureBundle:
    """Features plus the label, for one turn."""

    vector: np.ndarray
    label: int
    query_id: str
    user_id: str
    category: str = "general"


def build_features(question: str,
                   query_vector: np.ndarray,
                   sketch: MemorySketch,
                   probe: Optional[ProbeSignals] = None) -> np.ndarray:
    """Assemble the full pre-retrieval feature vector for one turn.

    `probe` may be None, which is how you run the sketch-only ablation without
    paying for any language model forward pass at all.
    """
    parts = [surface_features(question), sketch.features(query_vector)]
    parts.append(probe.as_vector() if probe is not None
                 else np.zeros(ProbeSignals.width(), dtype=np.float32))
    vector = np.concatenate(parts).astype(np.float32)
    if vector.shape[0] != FEATURE_WIDTH:
        raise AssertionError(
            f"feature width drifted: got {vector.shape[0]}, expected {FEATURE_WIDTH}. "
            "Update FEATURE_NAMES and GROUPS together when you add a signal.")
    return vector


def mask_groups(matrix: np.ndarray, keep: Sequence[str]) -> np.ndarray:
    """Zero out every feature group except the named ones.

    Used by the ablation in the paper's analysis section: `keep=["sketch"]`
    answers the question "how far does embedding geometry alone get us?".
    """
    unknown = set(keep) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")
    masked = np.zeros_like(matrix)
    for name in keep:
        columns = GROUPS[name]
        masked[:, columns] = matrix[:, columns]
    return masked


class FeatureScaler:
    """Standardise features using training-split statistics only."""

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None

    def fit(self, matrix: np.ndarray) -> "FeatureScaler":
        self.mean = matrix.mean(axis=0)
        self.scale = matrix.std(axis=0) + 1e-6
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("FeatureScaler.fit must be called before transform")
        return ((matrix - self.mean) / self.scale).astype(np.float32)

    def fit_transform(self, matrix: np.ndarray) -> np.ndarray:
        return self.fit(matrix).transform(matrix)

    def state_dict(self) -> dict:
        return {"mean": None if self.mean is None else self.mean.tolist(),
                "scale": None if self.scale is None else self.scale.tolist()}

    def load_state_dict(self, state: dict) -> "FeatureScaler":
        self.mean = None if state["mean"] is None else np.asarray(state["mean"], dtype=np.float32)
        self.scale = None if state["scale"] is None else np.asarray(state["scale"], dtype=np.float32)
        return self
