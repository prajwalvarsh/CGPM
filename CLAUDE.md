# CLAUDE.md

Read this before touching any file.

## What this project is

CGPM (Confidence-Gated Personal Memory). A small language model assistant has
a per-user long-term memory store. Before spending a retrieval call on that
store, a lightweight gate estimates whether retrieval will change the answer,
and routes the turn to one of three places: answer parametrically, retrieve
then answer, or ask for clarification.

Target: a workshop-length paper in about eight weeks. The paper draft lives
outside this repo, in the kit's `paper/` directory, and its tables map
one-to-one onto `experiments/log_template.md`.

## The invariant that defines this project

**No feature used by the gate at inference time may depend on retrieval
results.**

Everything in `src/signals.py` must be computable from: the raw text of the
turn, the query embedding, and the offline `MemorySketch`. If a change makes
a gate feature depend on `MemoryStore.search(...)`, that change is wrong even
if it improves the numbers, because it converts a pre-retrieval gate into a
post-retrieval reranker and deletes the contribution.

If you believe an exception is justified, stop and say so explicitly rather
than making the change.

## Layout

| Path | Role |
| --- | --- |
| `configs/default.yaml` | every hyper-parameter, no magic numbers in code |
| `src/config.py` | YAML loading, dotted `--set a.b=c` overrides |
| `src/data.py` | `MemoryItem`, `Query`, `UserBundle`; synthetic / JSONL / HF loaders |
| `src/memory.py` | `TextEncoder`, `MemoryStore` (expensive), `MemorySketch` (cheap) |
| `src/slm.py` | `SmallLM.answer`, `.probe`, `.verbalized`, `.clarify` |
| `src/signals.py` | the 19 features in three groups: surface, sketch, probe |
| `src/gate.py` | `ConfidenceGate`, temperature calibration, `select_thresholds` |
| `src/routing.py` | `run_policy` for never / always / verbalized / gate / oracle |
| `src/metrics.py` | quality, cost and calibration in one `summarize` row |
| `src/train_gate.py`, `src/evaluate.py` | the two CLI entry points |
| `scripts/smoke_test.py` | fast, offline, no model weights |

## Conventions

- Python 3.10 or newer. Type hints on public functions. `from __future__ import
  annotations` at the top of every module.
- Heavy imports (`torch`, `transformers`, `sentence_transformers`, `faiss`,
  `datasets`, `sklearn`) are **lazy**, inside the function or property that
  needs them. This is why `scripts/smoke_test.py` runs with no downloads.
  Preserve it.
- No em dashes in comments, docstrings or generated text.
- Every experiment writes to `experiments/runs/<run_name>/`. Never overwrite a
  previous run directory; pick a new `--run-name`.
- Splits are by **user**, never by query. `split_by_user` enforces this.
  Splitting by query leaks a user's memory store across train and test and
  silently inflates every number.
- New features go into an existing group in `src/signals.py`, and
  `FEATURE_NAMES` plus `GROUPS` must be updated together. `build_features`
  asserts on width drift so this fails loudly.
- Randomness comes from `config.seed`. Do not introduce an unseeded RNG.

## Definition of done for a change

1. `python scripts/smoke_test.py` passes.
2. `python -m src.train_gate --help` and `python -m src.evaluate --help` still
   work (this catches import errors without loading any model).
3. If the change affects numbers, a new row is added to
   `experiments/log_template.md` with what changed and what it concluded.
4. If the change adds a feature, the ablation table gains a row for it.

## Working style Prajwal wants

- **Small, reviewable diffs.** One idea per change. Do not refactor while
  fixing something.
- **Explain the why in a comment when a choice is non-obvious**, especially
  around cost. This repo's whole argument is about cost, so a line that quietly
  makes something more expensive needs a note.
- **Say when a result looks wrong.** A gate AUROC above 0.95 on a real
  benchmark almost certainly means label leakage, not success. Flag it rather
  than celebrating it.
- **Do not invent numbers.** If asked for a result that has not been run, say
  it has not been run.
- **Do not add dependencies** without saying why the standard library or an
  existing dependency will not do.

## Current state

The scaffold runs end to end on the synthetic corpus. The gate learns,
calibrates, saves and reloads. What is not done:

1. **Benchmark conversion.** `scripts/convert_benchmark.py` does not exist yet.
   It should read LaMP or LoCoMo and write the JSONL schema documented in
   `src/data.py:load_jsonl`.
2. **The `needs_memory` label for real data.** Implement the twice-run
   procedure: generate with and without retrieval, label positive where
   retrieval flips wrong to right, and record the reverse flips separately as
   evidence of retrieval harm.
3. **Baseline gates.** `verbalized` exists. SKR-style self-knowledge and a
   FLARE-style entropy trigger are described in the paper's setup section and
   still need implementing in `src/routing.py:score_turn`.
4. **Ablation driver.** `mask_groups` exists in `src/signals.py` but nothing
   calls it yet. A small script that trains the gate once per feature-group
   subset would fill Table 2 in one command.
5. **Conformal thresholds.** Replace or complement `select_thresholds` with a
   calibration that guarantees a user-specified miss rate.

Work items 1 and 2 are the critical path. Nothing else matters until the
project runs on a real benchmark.
