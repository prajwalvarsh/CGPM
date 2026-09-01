# cgpm-slm-memory

**Confidence-Gated Personal Memory: deciding whether to search a user's
long-term memory store before paying for retrieval, in a small language model.**

This is the starter repository for CGPM research
project. It combines the two topics you proposed into one experiment that can
be finished in eight weeks:

- **Retrieval confidence before RAG.** Estimate, cheaply and before the index
  is touched, whether retrieval will actually help this turn.
- **Personalized long-term memory for SLMs.** Make the thing being retrieved
  a per-user memory store rather than a static corpus.

They fit together because the personal setting is where the question gets
interesting. Whether a web corpus contains an answer is mostly a property of
the question. Whether *your* memory store contains it is a property of the
question *and* of a store that is small, idiosyncratic and growing. That is a
gating problem nobody has cleanly solved, and it is small enough to actually
finish.

## The claim you are testing

> A gate costing a small fraction of one retrieval call can match or exceed
> always-on retrieval while touching the memory store on well under half of
> user turns.

If that holds, you have a workshop paper. If it does not, the measured gap to
the oracle gate plus an honest cost-ratio sweep is *also* a workshop paper.
Both outcomes are publishable, which is why this project was chosen.

## What is in here

```
configs/default.yaml      every knob, overridable from the command line
src/config.py             config loading with dotted --set overrides
src/data.py               synthetic corpus, JSONL loader, HuggingFace loader
src/memory.py             per-user memory store, dense index, and the SKETCH
src/slm.py                small language model wrapper: answer, probe, verbalized
src/signals.py            the 19 pre-retrieval features, in three groups
src/gate.py               the confidence gate, calibration, threshold selection
src/routing.py            five routing policies through one shared code path
src/metrics.py            quality, cost and calibration metrics together
src/features_build.py     feature extraction and caching
src/train_gate.py         entry point: train and calibrate
src/evaluate.py           entry point: run policies, write the results table
scripts/smoke_test.py     no GPU, no network, no downloads: run this first
scripts/*.sh              the three runs you will repeat all project
experiments/log_template.md   the log whose rows map to the paper's tables
```

The single most important design rule: **nothing in `src/signals.py` may look
at retrieval results.** The moment a feature depends on what the index
returned, the gate is no longer a pre-retrieval gate and the contribution
evaporates. Keep that invariant.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

CPU is enough to get started. The synthetic corpus and the sketch-only gate
run on a laptop. You need a GPU only once you turn on the language model
probe and start generating answers.

## The first three things to run

**1. Prove the plumbing works (about 10 seconds, no downloads).**

```bash
python scripts/smoke_test.py
```

Every check should print `[ok]`. This exercises the config, the corpus, the
sketch, feature assembly, gate training, calibration, thresholds and the
metrics. If it passes, any later failure is in the model layer, which is a
much smaller place to look.

**2. Measure the gap you are competing for (needs the models, a few minutes).**

```bash
bash scripts/run_baselines.sh
```

This runs Never Retrieve, Always Retrieve and the oracle gate. Look at
`f1_memory_needed` versus `f1_general` in the output. You are looking for the
crossover: retrieval should help a lot on one and hurt on the other. **That
crossover is your entire research premise.** If it is not there, say so out
loud and adjust the setup (bigger store, harder general questions, smaller
model) before building anything on top.

**3. Train the gate and fill in the table.**

```bash
bash scripts/train_gate.sh     # trains twice: without and with the LM probe
bash scripts/run_full.sh       # every policy, one table
```

Copy `experiments/runs/main/results.md` into `experiments/log_template.md`.
That table is Table 1 of your paper.

## Where the real work is

The code here is a scaffold, not an answer. The parts that are genuinely
yours:

1. **Defining `needs_memory` on a real benchmark.** The synthetic corpus has
   an exact label. LaMP and LoCoMo do not. The recommended definition is to
   run the frozen generator twice, with and without retrieval, and label a
   turn positive where retrieval flips it from wrong to right. Write your
   definition down; it is a methodological choice reviewers will ask about.
2. **Converting a benchmark** into the JSONL schema in `src/data.py`. Put the
   converter in `scripts/convert_benchmark.py`.
3. **Better sketches.** Eight k-means centroids is the simplest thing that
   works. Coverage estimators, density scores and learned store embeddings are
   all open.
4. **Conformal thresholds.** Right now the threshold comes from a cost grid.
   Setting it to guarantee a user-specified miss rate is a real upgrade.

## Reading order for the literature

Start with three papers, in this order:

1. Mallen et al., *When Not to Trust Language Models* (arXiv:2212.10511). The
   empirical case that gating is worth doing at all.
2. Jeong et al., *Adaptive-RAG* (arXiv:2403.14403). The closest prior router.
   Note carefully what its classifier can and cannot see.
3. Salemi et al., *LaMP* (arXiv:2304.11406). The personalization benchmark you
   will be evaluated against.

The full list, with one-line takeaways, is in the kit's literature section.

## A note on honesty in the tables

Report answer quality, retrieval call rate and calibration together, always.
Each one alone is trivially gamed: never retrieving wins on cost, always
retrieving wins on quality, and a gate that outputs 0.5 everywhere has a
respectable-looking ECE. The contribution lives in the joint number, and
reviewers know it.
