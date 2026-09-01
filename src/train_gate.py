"""Entry point: train and calibrate the confidence gate.

    python -m src.train_gate --config configs/default.yaml

Outputs, all under `experiments/runs/<run_name>/`:
    gate.pt          the trained gate, its scaler and its temperatures
    thresholds.json  tau_low and tau_high chosen against the cost model
    train_report.json  gating metrics on the validation split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import add_config_args, config_from_args
from .data import load_dataset_splits
from .features_build import extract_split, save_features
from .gate import ConfidenceGate, save_thresholds, select_thresholds
from .memory import TextEncoder, build_stores
from .metrics import auroc, brier_score, expected_calibration_error
from .slm import SmallLM


def main() -> None:
    parser = argparse.ArgumentParser(description="train the CGPM confidence gate")
    add_config_args(parser)
    parser.add_argument("--run-name", default="gate")
    parser.add_argument("--no-probe", action="store_true",
                        help="skip language model probe features (sketch and "
                             "surface only, runs without a GPU)")
    parser.add_argument("--cost-missed", type=float, default=3.0,
                        help="cost of skipping a needed retrieval, relative to "
                             "the cost of an unnecessary one")
    args = parser.parse_args()
    config = config_from_args(args)

    run_dir = Path(config.get_path("eval.report_dir", "experiments/runs")) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("loading data")
    splits = load_dataset_splits(config)
    print("  " + json.dumps(splits.counts()))

    encoder = TextEncoder(
        model_name=config.get_path("memory.encoder_name"),
        batch_size=int(config.get_path("memory.embed_batch_size", 64)),
        normalize=bool(config.get_path("memory.normalize", True)),
    )
    sketch_clusters = int(config.get_path("memory.sketch_clusters", 8))
    seed = int(config.get("seed", 13))

    slm = None
    if not args.no_probe:
        slm = SmallLM(
            model_name=config.get_path("slm.model_name"),
            max_new_tokens=int(config.get_path("slm.max_new_tokens", 64)),
            probe_max_new_tokens=int(config.get_path("slm.probe_max_new_tokens", 8)),
            temperature=float(config.get_path("slm.temperature", 0.0)),
            dtype=str(config.get_path("slm.dtype", "auto")),
            device=str(config.get_path("slm.device", "auto")),
        )

    print("building memory stores")
    train_stores = build_stores(splits.train, encoder, sketch_clusters, seed)
    val_stores = build_stores(splits.val, encoder, sketch_clusters, seed)

    print("extracting features")
    x_train, y_train, users_train, ids_train = extract_split(
        splits.train, train_stores, encoder, slm, desc="train")
    x_val, y_val, users_val, ids_val = extract_split(
        splits.val, val_stores, encoder, slm, desc="val")
    save_features(run_dir / "features_train.npz", x_train, y_train, users_train, ids_train)
    save_features(run_dir / "features_val.npz", x_val, y_val, users_val, ids_val)
    print(f"  train {x_train.shape}, positives {float(y_train.mean()):.2f}")

    print("training gate")
    gate = ConfidenceGate(
        input_dim=x_train.shape[1],
        hidden_dim=int(config.get_path("gate.hidden_dim", 128)),
        dropout=float(config.get_path("gate.dropout", 0.1)),
        seed=seed,
    )
    gate.fit(x_train, y_train, x_val, y_val,
             epochs=int(config.get_path("gate.epochs", 30)),
             batch_size=int(config.get_path("gate.batch_size", 64)),
             lr=float(config.get_path("gate.lr", 1e-3)),
             weight_decay=float(config.get_path("gate.weight_decay", 1e-4)))

    print("calibrating")
    gate.calibrate(x_val, y_val, users_val,
                   per_user=bool(config.get_path("gate.per_user_calibration", True)))
    probabilities = gate.predict_proba(x_val, users_val)

    thresholds = select_thresholds(probabilities, y_val,
                                   cost_missed=args.cost_missed)
    print(f"  tau_low {thresholds.tau_low:.3f}  tau_high {thresholds.tau_high:.3f}")

    report = {
        "run_name": args.run_name,
        "counts": splits.counts(),
        "probe_features": not args.no_probe,
        "temperature": gate.temperature,
        "num_per_user_temperatures": len(gate.per_user_temperature),
        "val_auroc": auroc(probabilities, y_val),
        "val_ece": expected_calibration_error(probabilities, y_val),
        "val_brier": brier_score(probabilities, y_val),
        "tau_low": thresholds.tau_low,
        "tau_high": thresholds.tau_high,
        "cost_missed": args.cost_missed,
        "positive_rate_val": float(np.mean(y_val)) if len(y_val) else 0.0,
    }

    gate.save(run_dir / "gate.pt")
    save_thresholds(thresholds, run_dir / "thresholds.json")
    (run_dir / "train_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
