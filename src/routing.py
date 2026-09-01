"""The router: five policies over one shared code path.

Keeping every baseline inside a single `run_policy` function is deliberate.
It guarantees that "always retrieve" and "CGPM" differ only in the routing
decision and nothing else, so any gap in the results table is attributable to
the gate rather than to prompt drift between two implementations.

Policies:
    never       parametric answering only, never touch the memory store
    always      standard RAG, retrieve on every turn
    verbalized  ask the small model for a probability and threshold it
    gate        the trained confidence gate (this is CGPM)
    oracle      route using the ground-truth label; the ceiling, not a baseline
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .data import Query, UserBundle
from .gate import ConfidenceGate, GateThresholds
from .memory import MemoryStore, TextEncoder
from .signals import build_features
from .slm import SmallLM

POLICIES = ("never", "always", "verbalized", "gate", "oracle")


@dataclass
class TurnResult:
    query_id: str
    user_id: str
    question: str
    reference: str
    prediction: str
    route: str
    gate_score: float
    needs_memory: int
    retrieved_ids: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "query_id": self.query_id,
            "user_id": self.user_id,
            "question": self.question,
            "reference": self.reference,
            "prediction": self.prediction,
            "route": self.route,
            "gate_score": self.gate_score,
            "needs_memory": self.needs_memory,
            "retrieved_ids": self.retrieved_ids,
            "latency_ms": self.latency_ms,
        }


def score_turn(policy: str,
               query: Query,
               query_vector: np.ndarray,
               store: MemoryStore,
               slm: Optional[SmallLM],
               gate: Optional[ConfidenceGate],
               use_probe: bool = True) -> float:
    """Return P(memory helps) for one turn, without searching the index."""
    if policy == "never":
        return 0.0
    if policy == "always":
        return 1.0
    if policy == "oracle":
        return float(query.needs_memory)
    if policy == "verbalized":
        if slm is None:
            raise ValueError("the verbalized policy needs a language model")
        return slm.verbalized(query.question)
    if policy == "gate":
        if gate is None:
            raise ValueError("the gate policy needs a trained ConfidenceGate")
        probe = slm.probe(query.question) if (use_probe and slm is not None) else None
        features = build_features(query.question, query_vector, store.sketch, probe)
        return float(gate.predict_proba(features.reshape(1, -1),
                                        user_ids=[query.user_id])[0])
    raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")


def run_policy(policy: str,
               bundles: Sequence[UserBundle],
               stores: Dict[str, MemoryStore],
               encoder: TextEncoder,
               slm: SmallLM,
               gate: Optional[ConfidenceGate] = None,
               thresholds: Optional[GateThresholds] = None,
               top_k: int = 4,
               allow_clarify: bool = False,
               use_probe: bool = True,
               limit: Optional[int] = None) -> List[TurnResult]:
    """Run one policy end to end and return per-turn results."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    thresholds = thresholds or GateThresholds()

    results: List[TurnResult] = []
    for bundle in bundles:
        store = stores[bundle.user_id]
        questions = [q.question for q in bundle.queries]
        if not questions:
            continue
        query_vectors = encoder.encode(questions)

        for query, query_vector in zip(bundle.queries, query_vectors):
            if limit is not None and len(results) >= limit:
                return results

            started = time.perf_counter()
            probability = score_turn(policy, query, query_vector, store, slm,
                                     gate, use_probe)

            if policy == "never":
                route = "direct"
            elif policy == "always":
                route = "retrieve"
            elif allow_clarify:
                margin = float(store.sketch.features(query_vector)[2])
                route = thresholds.route_with_clarify(probability, margin)
            else:
                route = thresholds.route(probability)

            retrieved_ids: List[str] = []
            if route == "retrieve":
                hits = store.search(query_vector, top_k=top_k)
                retrieved_ids = [hit.item.memory_id for hit in hits]
                prediction = slm.answer(query.question,
                                        [hit.item.text for hit in hits])
            elif route == "clarify":
                prediction = slm.clarify(query.question)
            else:
                prediction = slm.answer(query.question, None)

            results.append(TurnResult(
                query_id=query.query_id,
                user_id=query.user_id,
                question=query.question,
                reference=query.answer,
                prediction=prediction,
                route=route,
                gate_score=probability,
                needs_memory=int(query.needs_memory),
                retrieved_ids=retrieved_ids,
                latency_ms=1000.0 * (time.perf_counter() - started),
            ))

    return results
