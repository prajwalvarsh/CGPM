"""A fast sanity check that needs no GPU, no downloads and no network.

    python scripts/smoke_test.py

It exercises the parts of the pipeline that do not require model weights:
config loading, the synthetic corpus, user-level splitting, the memory sketch,
feature assembly, gate training, calibration, threshold selection and the
metrics. If this passes, any later failure is in the model layer, which is a
much smaller place to look.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config                       # noqa: E402
from src.data import build_synthetic, split_by_user      # noqa: E402
from src.gate import ConfidenceGate, select_thresholds   # noqa: E402
from src.memory import build_sketch                      # noqa: E402
from src.metrics import (auroc, expected_calibration_error, risk_coverage_auc,
                         summarize, token_f1)            # noqa: E402
from src.signals import (FEATURE_WIDTH, GROUPS, build_features,
                         mask_groups, surface_features)  # noqa: E402

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "ok  " if condition else "FAIL"
    print(f"  [{status}] {name}{(' :: ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def main() -> int:
    print("config")
    config = load_config("configs/default.yaml", ["data.num_users=12", "gate.epochs=8"])
    check("yaml loads", config.get_path("memory.top_k") is not None)
    check("override applied", config.get_path("data.num_users") == 12,
          str(config.get_path("data.num_users")))

    print("data")
    bundles = build_synthetic(num_users=12, turns_per_user=40, queries_per_user=20,
                              memory_needed_rate=0.5, seed=13)
    check("users built", len(bundles) == 12)
    check("memories present", all(len(b.memories) > 0 for b in bundles))
    labels = [q.needs_memory for b in bundles for q in b.queries]
    positive_rate = float(np.mean(labels))
    check("label balance near 0.5", 0.3 < positive_rate < 0.7, f"{positive_rate:.2f}")

    again = build_synthetic(num_users=12, turns_per_user=40, queries_per_user=20,
                            seed=13)
    check("generator deterministic",
          [q.question for q in again[0].queries] == [q.question for q in bundles[0].queries])

    splits = split_by_user(bundles, (0.6, 0.2, 0.2), seed=13)
    train_users = {b.user_id for b in splits.train}
    test_users = {b.user_id for b in splits.test}
    check("splits disjoint by user", not (train_users & test_users))
    check("all users assigned",
          len(splits.train) + len(splits.val) + len(splits.test) == 12)

    print("memory sketch")
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(60, 32)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    sketch = build_sketch(vectors, num_clusters=6, seed=13)
    check("centroid count", sketch.centroids.shape[0] == 6, str(sketch.centroids.shape))
    check("occupancy sums to 1", abs(float(sketch.occupancy.sum()) - 1.0) < 1e-5)
    sketch_features = sketch.features(vectors[0])
    check("sketch feature width", sketch_features.shape == (8,), str(sketch_features.shape))
    check("sketch features finite", bool(np.isfinite(sketch_features).all()))

    empty = build_sketch(np.zeros((0, 32), dtype=np.float32))
    check("empty store is safe", empty.features(vectors[0]).shape == (8,))

    print("features")
    check("surface width", surface_features("What did I decide?").shape == (6,))
    vector = build_features("What did I decide about the thesis?", vectors[0], sketch)
    check("full feature width", vector.shape == (FEATURE_WIDTH,), str(vector.shape))
    check("probe columns zero without an LM",
          float(np.abs(vector[GROUPS["probe"]]).sum()) == 0.0)
    matrix = np.stack([vector, vector])
    masked = mask_groups(matrix, ["sketch"])
    check("group masking zeroes the rest",
          float(np.abs(masked[:, GROUPS["surface"]]).sum()) == 0.0)

    print("gate")
    n = 400
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, size=n)
    # A learnable signal in the first column, noise everywhere else.
    x = rng.normal(size=(n, FEATURE_WIDTH)).astype(np.float32)
    x[:, 0] += 2.5 * y
    users = [f"u{i % 8:02d}" for i in range(n)]

    gate = ConfidenceGate(input_dim=FEATURE_WIDTH, hidden_dim=32, seed=13)
    history = gate.fit(x[:300], y[:300], x[300:], y[300:], epochs=25,
                       batch_size=32, verbose=False)
    check("loss decreased", history["train_loss"][-1] < history["train_loss"][0],
          f"{history['train_loss'][0]:.3f} -> {history['train_loss'][-1]:.3f}")

    gate.calibrate(x[300:], y[300:], users[300:], per_user=True)
    probabilities = gate.predict_proba(x[300:], users[300:])
    check("probabilities in range",
          bool(((probabilities >= 0) & (probabilities <= 1)).all()))
    score = auroc(probabilities, y[300:])
    check("gate learns the signal", score > 0.8, f"auroc={score:.3f}")
    check("ece computable",
          np.isfinite(expected_calibration_error(probabilities, y[300:])))

    thresholds = select_thresholds(probabilities, y[300:], cost_missed=3.0)
    check("thresholds ordered", thresholds.tau_low < thresholds.tau_high,
          f"{thresholds.tau_low:.2f} < {thresholds.tau_high:.2f}")
    check("router returns known routes",
          thresholds.route(0.01) == "direct" and thresholds.route(0.99) == "retrieve")

    print("persistence")
    tmp = Path(".cache/smoke_gate.pt")
    gate.save(tmp)
    reloaded = ConfidenceGate.load(tmp)
    check("gate round-trips",
          bool(np.allclose(probabilities,
                           reloaded.predict_proba(x[300:], users[300:]), atol=1e-5)))
    tmp.unlink(missing_ok=True)

    print("metrics")
    check("token f1 exact", abs(token_f1("Tokyo", "tokyo.") - 1.0) < 1e-6)
    check("token f1 disjoint", token_f1("Paris", "Tokyo") == 0.0)
    row = summarize(
        predictions=["Tokyo", "You decided to postpone it", "no idea"],
        references=["Tokyo", "You decided to postpone it", "the marathon plan"],
        routes=["direct", "retrieve", "retrieve"],
        needs_memory=[0, 1, 1],
        gate_scores=[0.1, 0.9, 0.8],
        latencies_ms=[12.0, 140.0, 138.0],
    )
    check("row has quality", "f1" in row and 0 <= row["f1"] <= 100)
    check("row has cost", abs(row["retrieval_call_rate"] - 200 / 3) < 0.1,
          f"{row['retrieval_call_rate']:.1f}")
    check("row splits by need", "f1_memory_needed" in row and "f1_general" in row)
    check("risk-coverage finite",
          np.isfinite(risk_coverage_auc([0.9, 0.5, 0.1], [1.0, 1.0, 0.0])))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
