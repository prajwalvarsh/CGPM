"""A thin wrapper around a small instruction-tuned language model.

Three capabilities are needed by the rest of the pipeline:

    answer()      generate a response, optionally with retrieved memories
    probe()       a very short forward pass that yields uncertainty signals
                  BEFORE any retrieval happens
    verbalized()  ask the model directly how confident it is that it needs
                  the user's memory store

`probe` is the interesting one. It runs at most a handful of tokens, so its
cost is negligible next to a retrieval call, and it gives the gate access to
the model's own internal uncertainty rather than only to embedding geometry.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


ANSWER_SYSTEM = (
    "You are a helpful personal assistant. Answer in one short sentence. "
    "If you are given remembered notes about the user, rely on them. "
    "If you do not know, say so plainly."
)

PROBE_PROMPT = (
    "You are a personal assistant with a private memory of past conversations "
    "with this user. Before answering, decide whether you need to look "
    "something up in that memory. Reply with a single word, either MEMORY or "
    "GENERAL.\n\nUser turn: {question}\nDecision:"
)

VERBALIZED_PROMPT = (
    "You are a personal assistant with a private memory of past conversations "
    "with this user. Estimate the probability, between 0.00 and 1.00, that "
    "answering the turn below correctly requires looking in that memory. "
    "Reply with the number only.\n\nUser turn: {question}\nProbability:"
)


@dataclass
class ProbeSignals:
    """Uncertainty signals from a short forward pass, before retrieval."""

    first_token_entropy: float
    mean_token_entropy: float
    max_token_logprob: float
    margin: float               # top-1 minus top-2 probability on the first token
    memory_token_prob: float    # probability mass the model puts on "MEMORY"

    def as_vector(self) -> np.ndarray:
        return np.array([
            self.first_token_entropy,
            self.mean_token_entropy,
            self.max_token_logprob,
            self.margin,
            self.memory_token_prob,
        ], dtype=np.float32)

    @staticmethod
    def width() -> int:
        return 5


class SmallLM:
    """Loads lazily, so importing this module never downloads weights."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 max_new_tokens: int = 64, probe_max_new_tokens: int = 8,
                 temperature: float = 0.0, dtype: str = "auto",
                 device: str = "auto"):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.probe_max_new_tokens = probe_max_new_tokens
        self.temperature = temperature
        self.dtype = dtype
        self.device = device
        self._model = None
        self._tokenizer = None

    # ---------------- loading ----------------

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved_device = self.device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = "auto" if self.dtype == "auto" else getattr(torch, self.dtype)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch_dtype).to(resolved_device)
        self._model.eval()
        self.resolved_device = resolved_device

    @property
    def model(self):
        self._load()
        return self._model

    @property
    def tokenizer(self):
        self._load()
        return self._tokenizer

    # ---------------- prompting ----------------

    def _chat(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except (ValueError, AttributeError):
            # Base models without a chat template still need to work.
            return f"{system}\n\n{user}\n\nAnswer:"

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ---------------- public API ----------------

    def answer(self, question: str, memories: Optional[Sequence[str]] = None) -> str:
        """Answer a turn, with retrieved memories in context when supplied."""
        if memories:
            joined = "\n".join(f"- {m}" for m in memories)
            user = f"Remembered notes about this user:\n{joined}\n\nTurn: {question}"
        else:
            user = f"Turn: {question}"
        return self._generate(self._chat(ANSWER_SYSTEM, user), self.max_new_tokens)

    def clarify(self, question: str) -> str:
        """The abstain route: ask for the missing detail instead of guessing."""
        system = ("You are a personal assistant. You are not sure whether you "
                  "remember the detail the user is asking about. Ask one short "
                  "clarifying question, or say plainly that you do not have it.")
        return self._generate(self._chat(system, f"Turn: {question}"), 48)

    def probe(self, question: str) -> ProbeSignals:
        """Run a very short forward pass and read off uncertainty signals.

        Cost is a few tokens, which is why this may run on every turn while
        retrieval may not.
        """
        import torch

        prompt = PROBE_PROMPT.format(question=question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.probe_max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        entropies: List[float] = []
        max_logprobs: List[float] = []
        margin = 0.0
        for step, logits in enumerate(output.scores):
            probabilities = torch.softmax(logits[0].float(), dim=-1)
            log_probabilities = torch.log(probabilities + 1e-12)
            entropies.append(float(-(probabilities * log_probabilities).sum()))
            top2 = torch.topk(probabilities, k=2)
            max_logprobs.append(float(torch.log(top2.values[0] + 1e-12)))
            if step == 0:
                margin = float(top2.values[0] - top2.values[1])

        text = self.tokenizer.decode(
            output.sequences[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True).strip().upper()
        memory_token_prob = 1.0 if text.startswith("MEMORY") else 0.0

        return ProbeSignals(
            first_token_entropy=entropies[0] if entropies else 0.0,
            mean_token_entropy=float(np.mean(entropies)) if entropies else 0.0,
            max_token_logprob=max_logprobs[0] if max_logprobs else 0.0,
            margin=margin,
            memory_token_prob=memory_token_prob,
        )

    def verbalized(self, question: str) -> float:
        """Ask the model for a probability in words. The cheapest baseline gate.

        Following the finding that models can be asked directly for calibrated
        confidence, this is the policy `routing.policy=verbalized` uses.
        """
        text = self._generate(VERBALIZED_PROMPT.format(question=question), 8)
        match = re.search(r"(\d*\.?\d+)", text)
        if not match:
            return 0.5
        try:
            value = float(match.group(1))
        except ValueError:
            return 0.5
        if value > 1.0:                     # the model answered in percent
            value /= 100.0
        return float(min(max(value, 0.0), 1.0))


def entropy_of(probabilities: Sequence[float]) -> float:
    """Shannon entropy in nats, used by the metrics module and by tests."""
    total = sum(probabilities)
    if total <= 0:
        return 0.0
    return float(-sum((p / total) * math.log(p / total + 1e-12) for p in probabilities))
