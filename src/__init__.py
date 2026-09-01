"""CGPM: Confidence-Gated Personal Memory for small language models.

Package layout:
    config    configuration loading and dotted-key overrides
    data      personalized long-term memory datasets (synthetic and LaMP)
    memory    the per-user memory store, its dense index and its offline sketch
    slm       a thin wrapper around a small instruction-tuned language model
    signals   cheap pre-retrieval features consumed by the confidence gate
    gate      the confidence gate itself, plus calibration
    routing   the three-way router (direct / retrieve / clarify)
    metrics   task scores, cost accounting and calibration diagnostics
"""

__version__ = "0.1.0"
