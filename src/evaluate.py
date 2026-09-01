"""Entry point: run one or more routing policies and write a results row.

    python -m src.evaluate --policies never always gate --run-name main

Outputs, under `experiments/runs/<run_name>/`:
    results.json        one row per policy, the same fields as the paper table
    results.md          the same rows as a Markdown table you can paste into
                        experiments/log_template.md
    predictions_<p>.jsonl   per-turn records, when eval.save_predictions is true
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import add_config_args, config_from_args
from .data import load_dataset_splits
from .gate import ConfidenceGate, GateThresholds, load_thresholds
from .memory import TextEncoder, build_stores
from .metrics import summarize
from .routing import POLICIES, run_policy
from .slm import SmallLM

REPORT_COLUMNS = [
    "policy", "n", "f1", "em", "f1_memory_needed", "f1_general",
    "retrieval_call_rate", "unnecessary_retrieval_rate", "missed_retrieval_rate",
    "clarify_rate", "gate_auroc", "gate_ece", "latency_ms_mean",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="evaluate CGPM routing policies")
    add_config_args(parser)
    parser.add_argument("--policies", nargs="+", default=["never", "always", "gate"],
                        choices=list(POLICIES))
    parser.add_argument("--gate-dir", default=None,
                        help="directory holding gate.pt and thresholds.json; "
                             "defaults to <report_dir>/gate")
    parser.add_argument("--run-name", default="main")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--allow-clarify", action="store_true",
                        help="enable the three-way router with abstention")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N turns, for a quick smoke run")
    args = parser.parse_args()
    config = config_from_args(args)

    report_dir = Path(config.get_path("eval.report_dir", "experiments/runs"))
    run_dir = report_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    splits = load_dataset_splits(config)
    bundles = getattr(splits, args.split)
    print(f"evaluating on {args.split}: {len(bundles)} users, "
          f"{sum(len(b.queries) for b in bundles)} turns")

    encoder = TextEncoder(
        model_name=config.get_path("memory.encoder_name"),
        batch_size=int(config.get_path("memory.embed_batch_size", 64)),
        normalize=bool(config.get_path("memory.normalize", True)),
    )
    slm = SmallLM(
        model_name=config.get_path("slm.model_name"),
        max_new_tokens=int(config.get_path("slm.max_new_tokens", 64)),
        probe_max_new_tokens=int(config.get_path("slm.probe_max_new_tokens", 8)),
        temperature=float(config.get_path("slm.temperature", 0.0)),
        dtype=str(config.get_path("slm.dtype", "auto")),
        device=str(config.get_path("slm.device", "auto")),
    )
    stores = build_stores(bundles, encoder,
                          int(config.get_path("memory.sketch_clusters", 8)),
                          int(config.get("seed", 13)))

    gate = None
    thresholds = GateThresholds(
        tau_low=float(config.get_path("gate.tau_low", 0.35)),
        tau_high=float(config.get_path("gate.tau_high", 0.85)),
    )
    if "gate" in args.policies:
        gate_dir = Path(args.gate_dir) if args.gate_dir else report_dir / "gate"
        gate_path = gate_dir / "gate.pt"
        if not gate_path.exists():
            raise FileNotFoundError(
                f"no trained gate at {gate_path}. Run `python -m src.train_gate` "
                "first, or drop 'gate' from --policies.")
        gate = ConfidenceGate.load(gate_path)
        threshold_path = gate_dir / "thresholds.json"
        if threshold_path.exists():
            thresholds = load_thresholds(threshold_path)
        print(f"loaded gate from {gate_path}, "
              f"tau_low={thresholds.tau_low:.3f}")

    rows = []
    for policy in args.policies:
        print(f"\nrunning policy: {policy}")
        results = run_policy(
            policy, bundles, stores, encoder, slm, gate, thresholds,
            top_k=int(config.get_path("memory.top_k", 4)),
            allow_clarify=args.allow_clarify,
            limit=args.limit,
        )
        row = summarize(
            predictions=[r.prediction for r in results],
            references=[r.reference for r in results],
            routes=[r.route for r in results],
            needs_memory=[r.needs_memory for r in results],
            gate_scores=[r.gate_score for r in results],
            latencies_ms=[r.latency_ms for r in results],
        )
        row["policy"] = policy
        rows.append(row)
        print("  " + json.dumps({k: round(v, 2) for k, v in row.items()
                                 if isinstance(v, float)}))

        if bool(config.get_path("eval.save_predictions", True)):
            with (run_dir / f"predictions_{policy}.jsonl").open("w", encoding="utf-8") as fh:
                for result in results:
                    fh.write(json.dumps(result.to_dict()) + "\n")

    (run_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    columns = [c for c in REPORT_COLUMNS if any(c in row for row in rows)]
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            cells.append(f"{value:.1f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    (run_dir / "results.md").write_text(table + "\n", encoding="utf-8")

    print("\n" + table)
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
