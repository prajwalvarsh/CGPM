# Experiment log

One row per run. The columns are deliberately the same as Table 1 of
`paper/main.tex`, so filling this file in is literally filling in the paper.
Never delete a row, including the ones that failed. The failed rows are what
you will need when a reviewer asks why you did not try the obvious thing.

Copy `experiments/runs/<run_name>/results.md` straight into the table below,
then add the two columns a script cannot fill in for you: what you changed and
what you concluded.

---

## Table 1: main results

Cost ratio `c_fn / c_fp`: **3** (state it every time, it determines the
operating point).
Generator: **Qwen2.5-1.5B-Instruct**. Encoder: **all-MiniLM-L6-v2**. Seeds: **3**.

| Run ID | Date | Policy | Dataset | Task score (F1) | F1 memory-needed | F1 general | Calls (%) | Unnecessary (%) | Missed (%) | Gate AUROC | ECE | Latency (ms) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r001 | | never | synthetic | | | | 0.0 | 0.0 | 100.0 | n/a | n/a | | parametric floor |
| r002 | | always | synthetic | | | | 100.0 | 100.0 | 0.0 | n/a | n/a | | standard RAG ceiling on cost |
| r003 | | oracle | synthetic | | | | | 0.0 | 0.0 | 1.00 | 0.00 | | quality ceiling, not a baseline |
| r004 | | verbalized | synthetic | | | | | | | | | | training-free gate |
| r005 | | gate | synthetic | | | | | | | | | | CGPM, first working version |
| r006 | | gate | LoCoMo | | | | | | | | | | first real benchmark |

**How to read this table.** A row is only interesting if the pair
(task score, calls) moves. A gate that improves score by retrieving more has
not demonstrated anything: compare it against `always` at the same cost, not
against `never`.

---

## Table 2: feature-family ablation

Fill this by re-training the gate with `mask_groups` applied
(`src/signals.py`). Every row uses the same generator and the same split.

| Run ID | Feature families | Gate AUROC | ECE | Calls (%) | Task F1 | Cost vs one retrieval | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a001 | surface only | | | | | <0.01 | the free floor |
| a002 | sketch only | | | | | | is embedding geometry enough? |
| a003 | probe only | | | | | | is the model's own uncertainty enough? |
| a004 | surface + sketch | | | | | | the no-GPU configuration |
| a005 | sketch + probe | | | | | | |
| a006 | all three | | | | | | CGPM as reported |

---

## Table 3: cost-ratio sweep

| Run ID | c_fn / c_fp | tau_low | Calls (%) | Unnecessary (%) | Missed (%) | Task F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c001 | 1:1 | | | | | | |
| c002 | 2:1 | | | | | | |
| c003 | 3:1 | | | | | | default |
| c004 | 5:1 | | | | | | |
| c005 | 10:1 | | | | | | approaches always-retrieve |

---

## Run journal

One short entry per working session. Two or three sentences. The point is not
documentation, it is that in week 6 you will not remember why you abandoned
something in week 2.

### r001, YYYY-MM-DD
- **Changed:**
- **Command:**
- **Result:**
- **Concluded:**
- **Next:**

### r002, YYYY-MM-DD
- **Changed:**
- **Command:**
- **Result:**
- **Concluded:**
- **Next:**

---

## Things that did not work

Keep this section. It becomes the limitations section of the paper, and it
stops you from re-running the same dead end in week 7.

| Date | What you tried | Why you expected it to work | What actually happened |
| --- | --- | --- | --- |
| | | | |
