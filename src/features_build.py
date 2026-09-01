"""Build the feature matrix the confidence gate trains on.

This is separated from `train_gate.py` because feature extraction is the
expensive part (it touches the encoder and, optionally, the small language
model) while training the gate itself takes seconds. Cache the matrix once,
then sweep gate hyper-parameters and ablations for free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .data import UserBundle
from .memory import MemoryStore, TextEncoder
from .signals import FEATURE_NAMES, build_features
from .slm import SmallLM


def extract_split(bundles: Sequence[UserBundle],
                  stores: Dict[str, MemoryStore],
                  encoder: TextEncoder,
                  slm: Optional[SmallLM] = None,
                  desc: str = "features") -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Return (features, labels, user_ids, query_ids) for one split.

    Pass `slm=None` to skip the probe features entirely. That path needs no
    GPU and is a useful sanity run before you commit to a full extraction.
    """
    rows: List[np.ndarray] = []
    labels: List[int] = []
    user_ids: List[str] = []
    query_ids: List[str] = []

    for bundle in tqdm(list(bundles), desc=desc, unit="user"):
        store = stores[bundle.user_id]
        questions = [q.question for q in bundle.queries]
        if not questions:
            continue
        query_vectors = encoder.encode(questions)
        for query, query_vector in zip(bundle.queries, query_vectors):
            probe = slm.probe(query.question) if slm is not None else None
            rows.append(build_features(query.question, query_vector,
                                       store.sketch, probe))
            labels.append(int(query.needs_memory))
            user_ids.append(query.user_id)
            query_ids.append(query.query_id)

    features = (np.stack(rows).astype(np.float32) if rows
                else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32))
    return features, np.asarray(labels, dtype=np.int64), user_ids, query_ids


def save_features(path: str | Path, features: np.ndarray, labels: np.ndarray,
                  user_ids: Sequence[str], query_ids: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features, labels=labels,
                        user_ids=np.asarray(user_ids, dtype=object),
                        query_ids=np.asarray(query_ids, dtype=object),
                        feature_names=np.asarray(FEATURE_NAMES, dtype=object))


def load_features(path: str | Path):
    payload = np.load(Path(path), allow_pickle=True)
    return (payload["features"], payload["labels"],
            list(payload["user_ids"]), list(payload["query_ids"]))
