"""The confidence gate.

A two-layer MLP over the pre-retrieval features that outputs
p_hat = P(searching this user's memory store changes the answer).

Two things beyond a plain classifier matter here, and both are worth a
paragraph in the paper:

1. Calibration. A gate is only useful if p_hat means something. Temperature
   scaling on the validation split is the standard fix and costs one scalar.
   `per_user_temperature` goes further and fits one scalar per user, which is
   the personalization angle applied to the gate itself rather than to the
   answer.
2. Threshold selection. `tau_low` and `tau_high` should be chosen against an
   explicit cost model (what does one unnecessary retrieval cost relative to
   one missed one?), not tuned by eye. `select_thresholds` does that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .signals import FEATURE_WIDTH, FeatureScaler


@dataclass
class GateThresholds:
    tau_low: float = 0.35
    tau_high: float = 0.85

    def route(self, probability: float) -> str:
        """Two-way routing: retrieve unless the gate is confident it need not."""
        return "direct" if probability < self.tau_low else "retrieve"

    def route_with_clarify(self, probability: float, margin: float) -> str:
        """Routing that can also abstain.

        The clarify route fires when the gate is confident that memory is
        needed but the sketch says the store does not cover the query, which
        is the situation where retrieval will return something plausible and
        wrong. `margin` is the sketch centroid margin for the turn.
        """
        if probability < self.tau_low:
            return "direct"
        if probability > self.tau_high and margin < 0.0:
            return "clarify"
        return "retrieve"


class ConfidenceGate:
    """Two-layer MLP with temperature calibration."""

    def __init__(self, input_dim: int = FEATURE_WIDTH, hidden_dim: int = 128,
                 dropout: float = 0.1, seed: int = 13):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.seed = seed
        self.scaler = FeatureScaler()
        self.temperature: float = 1.0
        self.per_user_temperature: Dict[str, float] = {}
        self._net = None

    # ---------------- construction ----------------

    def _build(self):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    @property
    def net(self):
        if self._net is None:
            self._net = self._build()
        return self._net

    # ---------------- training ----------------

    def fit(self, features: np.ndarray, labels: np.ndarray,
            val_features: Optional[np.ndarray] = None,
            val_labels: Optional[np.ndarray] = None,
            epochs: int = 30, batch_size: int = 64, lr: float = 1e-3,
            weight_decay: float = 1e-4, verbose: bool = True) -> Dict[str, list]:
        """Train with class-balanced BCE. Returns a per-epoch history."""
        import torch
        from torch import nn

        scaled = self.scaler.fit_transform(features)
        x = torch.tensor(scaled, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)

        positive_rate = float(labels.mean()) if len(labels) else 0.5
        positive_weight = torch.tensor(
            [(1.0 - positive_rate) / max(positive_rate, 1e-6)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr,
                                      weight_decay=weight_decay)

        history: Dict[str, list] = {"train_loss": [], "val_loss": []}
        num_samples = x.shape[0]
        generator = torch.Generator().manual_seed(self.seed)

        for epoch in range(epochs):
            self.net.train()
            permutation = torch.randperm(num_samples, generator=generator)
            epoch_loss = 0.0
            for start in range(0, num_samples, batch_size):
                batch = permutation[start:start + batch_size]
                optimizer.zero_grad()
                loss = criterion(self.net(x[batch]), y[batch])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item() * len(batch)
            train_loss = epoch_loss / max(num_samples, 1)
            history["train_loss"].append(train_loss)

            val_loss = float("nan")
            if val_features is not None and val_labels is not None:
                self.net.eval()
                with torch.no_grad():
                    vx = torch.tensor(self.scaler.transform(val_features),
                                      dtype=torch.float32)
                    vy = torch.tensor(val_labels, dtype=torch.float32).reshape(-1, 1)
                    val_loss = float(criterion(self.net(vx), vy))
            history["val_loss"].append(val_loss)

            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"  epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}")

        return history

    # ---------------- inference ----------------

    def logits(self, features: np.ndarray) -> np.ndarray:
        import torch

        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(self.scaler.transform(features), dtype=torch.float32)
            return self.net(x).reshape(-1).numpy()

    def predict_proba(self, features: np.ndarray,
                      user_ids: Optional[Sequence[str]] = None) -> np.ndarray:
        """Calibrated P(memory helps) for each row."""
        raw = self.logits(features)
        if user_ids is None:
            temperatures = np.full(raw.shape, self.temperature, dtype=np.float32)
        else:
            temperatures = np.array(
                [self.per_user_temperature.get(u, self.temperature) for u in user_ids],
                dtype=np.float32)
        return 1.0 / (1.0 + np.exp(-raw / np.maximum(temperatures, 1e-3)))

    # ---------------- calibration ----------------

    def calibrate(self, features: np.ndarray, labels: np.ndarray,
                  user_ids: Optional[Sequence[str]] = None,
                  per_user: bool = False) -> None:
        """Fit temperature(s) on a held-out split by grid search on NLL."""
        raw = self.logits(features)
        self.temperature = _best_temperature(raw, labels)
        self.per_user_temperature = {}
        if per_user and user_ids is not None:
            user_ids = np.asarray(user_ids)
            for user in np.unique(user_ids):
                mask = user_ids == user
                # Fall back to the global temperature when a user is too sparse
                # to estimate one reliably.
                if int(mask.sum()) >= 12 and 0 < labels[mask].mean() < 1:
                    self.per_user_temperature[str(user)] = _best_temperature(
                        raw[mask], labels[mask])

    # ---------------- persistence ----------------

    def save(self, path: str | Path) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "config": {"input_dim": self.input_dim, "hidden_dim": self.hidden_dim,
                       "dropout": self.dropout, "seed": self.seed},
            "scaler": self.scaler.state_dict(),
            "temperature": self.temperature,
            "per_user_temperature": self.per_user_temperature,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "ConfidenceGate":
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        gate = cls(**payload["config"])
        gate.net.load_state_dict(payload["state_dict"])
        gate.scaler.load_state_dict(payload["scaler"])
        gate.temperature = float(payload["temperature"])
        gate.per_user_temperature = dict(payload["per_user_temperature"])
        return gate


def _best_temperature(raw_logits: np.ndarray, labels: np.ndarray) -> float:
    """Grid search a scalar temperature minimising negative log likelihood."""
    best_temperature, best_nll = 1.0, float("inf")
    for temperature in np.linspace(0.25, 5.0, 96):
        probabilities = 1.0 / (1.0 + np.exp(-raw_logits / temperature))
        probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
        nll = float(-np.mean(labels * np.log(probabilities)
                             + (1 - labels) * np.log(1 - probabilities)))
        if nll < best_nll:
            best_temperature, best_nll = float(temperature), nll
    return best_temperature


def select_thresholds(probabilities: np.ndarray, labels: np.ndarray,
                      cost_unnecessary: float = 1.0,
                      cost_missed: float = 3.0,
                      grid: int = 50) -> GateThresholds:
    """Pick tau_low and tau_high against an explicit cost model.

    `cost_unnecessary` is what you pay for retrieving when memory was not
    needed (latency, tokens, distraction). `cost_missed` is what you pay for
    skipping retrieval when it was needed (a wrong personal answer). The
    default 1:3 ratio says a wrong personal fact hurts three times as much as
    a wasted lookup. Report whichever ratio you use: it fully determines the
    operating point, and reviewers will ask.
    """
    candidates = np.linspace(0.05, 0.95, grid)
    best = GateThresholds()
    best_cost = float("inf")
    for tau_low in candidates:
        retrieved = probabilities >= tau_low
        unnecessary = int(np.sum(retrieved & (labels == 0)))
        missed = int(np.sum(~retrieved & (labels == 1)))
        cost = cost_unnecessary * unnecessary + cost_missed * missed
        if cost < best_cost:
            best_cost = cost
            best = GateThresholds(tau_low=float(tau_low),
                                  tau_high=float(min(0.99, tau_low + 0.5)))
    return best


def save_thresholds(thresholds: GateThresholds, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(thresholds), indent=2), encoding="utf-8")


def load_thresholds(path: str | Path) -> GateThresholds:
    return GateThresholds(**json.loads(Path(path).read_text(encoding="utf-8")))
