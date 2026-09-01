"""Personalized long-term memory datasets.

Every dataset in this repo is reduced to the same three objects:

    MemoryItem   one thing the assistant remembers about one user
    Query        one user turn, with a gold answer and a `needs_memory` label
    UserBundle   all memories plus all queries for a single user

`needs_memory` is the supervision signal for the confidence gate. It answers
the question the gate has to answer before retrieval runs: would searching
this user's memory store actually change the answer?

Three sources are supported.

1. `synthetic`  A deterministic generator. No downloads, runs on a laptop in
   seconds, and gives you a clean `needs_memory` label. Use it to get the
   whole loop working before you touch a real benchmark.
2. `jsonl`      Your own file in the schema documented in `load_jsonl`.
   This is the path you will use once you have converted LaMP, LongLaMP,
   LoCoMo, PerLTQA or MSC into this repo's format.
3. `hf`         Any HuggingFace dataset, with a field mapping supplied in the
   config so you do not have to edit this file to try a new benchmark.

Converting a public benchmark is deliberately left as your work: it is the
first real research decision in the project, because how you define
`needs_memory` on a benchmark that was not built for it is a contribution in
its own right. `docs/` in your write-up should record that definition.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------
# Core records
# --------------------------------------------------------------------------

@dataclass
class MemoryItem:
    """One entry in a user's personal long-term memory store."""

    memory_id: str
    user_id: str
    text: str
    kind: str = "episodic"       # episodic | summary | fact
    session: int = 0             # which conversation session it came from

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Query:
    """One user turn to answer."""

    query_id: str
    user_id: str
    question: str
    answer: str
    needs_memory: bool
    gold_memory_ids: List[str] = field(default_factory=list)
    category: str = "general"    # free-form tag, useful for the analysis section

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class UserBundle:
    user_id: str
    memories: List[MemoryItem]
    queries: List[Query]


@dataclass
class DatasetSplits:
    train: List[UserBundle]
    val: List[UserBundle]
    test: List[UserBundle]

    def counts(self) -> Dict[str, int]:
        return {
            "train_users": len(self.train),
            "val_users": len(self.val),
            "test_users": len(self.test),
            "train_queries": sum(len(u.queries) for u in self.train),
            "val_queries": sum(len(u.queries) for u in self.val),
            "test_queries": sum(len(u.queries) for u in self.test),
        }


# --------------------------------------------------------------------------
# Synthetic generator
# --------------------------------------------------------------------------

_PROJECTS = ["the thesis proposal", "the internship application",
             "the reading group", "the kitchen renovation",
             "the marathon training plan", "the open-source patch",
             "the visa paperwork", "the guitar lessons"]

_PEOPLE = ["Dr. Rao", "Anika", "Marcus", "the landlord", "my sister",
           "the lab manager", "Priya", "the recruiter"]

_DECISIONS = ["switch the topic to retrieval calibration",
              "postpone it until after the exams",
              "split the work into two shorter phases",
              "use the smaller model to keep latency down",
              "book the earlier slot",
              "drop the third experiment",
              "share the draft before the deadline",
              "pay the fee in two instalments"]

_PREFERENCES = ["I take my coffee without sugar",
                "I read papers on the train, never at my desk",
                "I prefer written summaries over meetings",
                "I run in the morning, not in the evening",
                "I keep Fridays free of calls",
                "I write notes in Markdown, not in a notebook",
                "I dislike open-plan offices",
                "I always book aisle seats"]

_SMALL_TALK = [
    ("What is the capital of Japan?", "Tokyo."),
    ("How many minutes are in a day?", "1440."),
    ("What does GPU stand for?", "Graphics processing unit."),
    ("Give me a synonym for 'concise'.", "Terse."),
    ("What is 15 percent of 200?", "30."),
    ("Which ocean is the largest?", "The Pacific Ocean."),
    ("What language is spoken in Brazil?", "Portuguese."),
    ("Round 3.14159 to two decimals.", "3.14."),
    ("What is the boiling point of water at sea level?",
     "100 degrees Celsius."),
    ("Name a primary colour.", "Red."),
]


def build_synthetic(num_users: int = 40,
                    turns_per_user: int = 60,
                    queries_per_user: int = 25,
                    memory_needed_rate: float = 0.5,
                    seed: int = 13) -> List[UserBundle]:
    """Build a deterministic personalized-memory corpus.

    Each user gets a store of episodic notes, stable preferences and session
    summaries. Roughly `memory_needed_rate` of their queries can only be
    answered from that store; the rest are answerable from the model's own
    parameters and are exactly the turns a good gate should refuse to
    retrieve for.
    """
    rng = random.Random(seed)
    bundles: List[UserBundle] = []

    for user_index in range(num_users):
        user_id = f"u{user_index:03d}"
        memories: List[MemoryItem] = []
        answerable: List[Tuple[str, str, str]] = []  # (question, answer, mem_id)

        project_pool = rng.sample(_PROJECTS, k=min(4, len(_PROJECTS)))
        pref_pool = rng.sample(_PREFERENCES, k=min(3, len(_PREFERENCES)))

        counter = 0
        for session in range(max(1, turns_per_user // 10)):
            for _ in range(10):
                if counter >= turns_per_user:
                    break
                memory_id = f"{user_id}-m{counter:03d}"
                roll = rng.random()
                if roll < 0.45:
                    project = rng.choice(project_pool)
                    decision = rng.choice(_DECISIONS)
                    person = rng.choice(_PEOPLE)
                    text = (f"In session {session}, I talked to {person} about "
                            f"{project} and decided to {decision}.")
                    memories.append(MemoryItem(memory_id, user_id, text,
                                               "episodic", session))
                    answerable.append((
                        f"What did I decide about {project}?",
                        f"You decided to {decision}, after talking to {person}.",
                        memory_id))
                elif roll < 0.75:
                    preference = rng.choice(pref_pool)
                    text = f"User preference noted in session {session}: {preference}."
                    memories.append(MemoryItem(memory_id, user_id, text,
                                               "fact", session))
                    answerable.append((
                        "Remind me of one habit I told you about.",
                        preference.capitalize() + ".",
                        memory_id))
                else:
                    project = rng.choice(project_pool)
                    text = (f"Summary of session {session}: we mostly worked on "
                            f"{project} and left the next step open.")
                    memories.append(MemoryItem(memory_id, user_id, text,
                                               "summary", session))
                    answerable.append((
                        f"What was session {session} mostly about?",
                        f"Mostly {project}.",
                        memory_id))
                counter += 1

        num_memory_queries = int(round(queries_per_user * memory_needed_rate))
        num_general_queries = queries_per_user - num_memory_queries

        queries: List[Query] = []
        chosen = rng.sample(answerable, k=min(num_memory_queries, len(answerable)))
        for q_index, (question, answer, memory_id) in enumerate(chosen):
            queries.append(Query(
                query_id=f"{user_id}-q{q_index:03d}",
                user_id=user_id,
                question=question,
                answer=answer,
                needs_memory=True,
                gold_memory_ids=[memory_id],
                category="personal",
            ))

        offset = len(queries)
        for q_index in range(num_general_queries):
            question, answer = _SMALL_TALK[(user_index + q_index) % len(_SMALL_TALK)]
            queries.append(Query(
                query_id=f"{user_id}-q{offset + q_index:03d}",
                user_id=user_id,
                question=question,
                answer=answer,
                needs_memory=False,
                gold_memory_ids=[],
                category="general",
            ))

        rng.shuffle(queries)
        bundles.append(UserBundle(user_id, memories, queries))

    return bundles


# --------------------------------------------------------------------------
# File and hub loaders
# --------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> List[UserBundle]:
    """Read a corpus from JSONL, one user per line.

    Expected schema per line::

        {"user_id": "u001",
         "memories": [{"memory_id": "...", "text": "...", "kind": "episodic",
                       "session": 0}, ...],
         "queries":  [{"query_id": "...", "question": "...", "answer": "...",
                       "needs_memory": true, "gold_memory_ids": ["..."],
                       "category": "personal"}, ...]}

    This is the format `scripts/convert_benchmark.py` should write when you
    adapt LaMP, LoCoMo or PerLTQA. Keeping one schema means every policy,
    metric and plot in this repo keeps working when you change benchmark.
    """
    path = Path(path)
    bundles: List[UserBundle] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            user_id = str(record["user_id"])
            memories = [
                MemoryItem(
                    memory_id=str(m.get("memory_id", f"{user_id}-m{i:04d}")),
                    user_id=user_id,
                    text=str(m["text"]),
                    kind=str(m.get("kind", "episodic")),
                    session=int(m.get("session", 0)),
                )
                for i, m in enumerate(record.get("memories", []))
            ]
            queries = [
                Query(
                    query_id=str(q.get("query_id", f"{user_id}-q{i:04d}")),
                    user_id=user_id,
                    question=str(q["question"]),
                    answer=str(q.get("answer", "")),
                    needs_memory=bool(q["needs_memory"]),
                    gold_memory_ids=[str(g) for g in q.get("gold_memory_ids", [])],
                    category=str(q.get("category", "general")),
                )
                for i, q in enumerate(record.get("queries", []))
            ]
            bundles.append(UserBundle(user_id, memories, queries))
    return bundles


def load_hf(repo: str,
            name: Optional[str] = None,
            split: str = "train",
            fields: Optional[Dict[str, str]] = None,
            cache_dir: Optional[str] = None) -> List[UserBundle]:
    """Load any HuggingFace dataset through a field mapping.

    `fields` maps this repo's names onto the dataset's column names, for
    example::

        {"user_id": "user", "question": "input", "answer": "output",
         "memories": "profile", "memory_text": "text"}

    Rows sharing a `user_id` are grouped into one `UserBundle`. `needs_memory`
    is not present in most public benchmarks, so it is left as True here and
    you should overwrite it with your own definition in the converter. Write
    that definition down: it is a methodological choice reviewers will ask
    about.
    """
    from datasets import load_dataset  # imported lazily so the repo imports offline

    fields = dict(fields or {})
    key_user = fields.get("user_id", "user_id")
    key_question = fields.get("question", "question")
    key_answer = fields.get("answer", "answer")
    key_memories = fields.get("memories", "profile")
    key_memory_text = fields.get("memory_text", "text")

    dataset = load_dataset(repo, name, split=split, cache_dir=cache_dir)

    grouped: Dict[str, UserBundle] = {}
    for index, row in enumerate(dataset):
        user_id = str(row.get(key_user, index))
        bundle = grouped.setdefault(user_id, UserBundle(user_id, [], []))
        raw_memories = row.get(key_memories) or []
        if bundle.memories == [] and raw_memories:
            for m_index, entry in enumerate(raw_memories):
                text = entry.get(key_memory_text) if isinstance(entry, dict) else str(entry)
                if not text:
                    continue
                bundle.memories.append(MemoryItem(
                    memory_id=f"{user_id}-m{m_index:04d}",
                    user_id=user_id,
                    text=str(text),
                ))
        bundle.queries.append(Query(
            query_id=f"{user_id}-q{len(bundle.queries):04d}",
            user_id=user_id,
            question=str(row.get(key_question, "")),
            answer=str(row.get(key_answer, "")),
            needs_memory=True,
            gold_memory_ids=[],
            category="benchmark",
        ))
    return list(grouped.values())


# --------------------------------------------------------------------------
# Splitting and the single entry point used by the scripts
# --------------------------------------------------------------------------

def split_by_user(bundles: Sequence[UserBundle],
                  ratios: Sequence[float] = (0.6, 0.2, 0.2),
                  seed: int = 13) -> DatasetSplits:
    """Split at the user level, never at the query level.

    Splitting by query would let the gate see the same person's memory store
    at train and test time, which quietly inflates every number you report.
    """
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must be three numbers summing to 1, got {ratios}")
    ordered = sorted(bundles, key=lambda b: b.user_id)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_total = len(ordered)
    n_train = int(round(ratios[0] * n_total))
    n_val = int(round(ratios[1] * n_total))
    return DatasetSplits(
        train=ordered[:n_train],
        val=ordered[n_train:n_train + n_val],
        test=ordered[n_train + n_val:],
    )


def load_dataset_splits(config) -> DatasetSplits:
    """Build train/val/test splits from the `data` block of a config."""
    data_cfg = config["data"]
    source = str(data_cfg.get("dataset", "synthetic")).lower()

    if source == "synthetic":
        bundles = build_synthetic(
            num_users=int(data_cfg.get("num_users", 40)),
            turns_per_user=int(data_cfg.get("turns_per_user", 60)),
            queries_per_user=int(data_cfg.get("queries_per_user", 25)),
            memory_needed_rate=float(data_cfg.get("memory_needed_rate", 0.5)),
            seed=int(config.get("seed", 13)),
        )
    elif source == "jsonl":
        bundles = load_jsonl(data_cfg["jsonl_path"])
    elif source == "hf":
        bundles = load_hf(
            repo=data_cfg["hf_repo"],
            name=data_cfg.get("hf_name"),
            split=data_cfg.get("hf_split", "train"),
            fields=data_cfg.get("hf_fields"),
            cache_dir=data_cfg.get("cache_dir"),
        )
    else:
        raise ValueError(
            f"unknown data.dataset {source!r}; expected synthetic, jsonl or hf")

    return split_by_user(bundles,
                         ratios=tuple(data_cfg.get("split_ratios", (0.6, 0.2, 0.2))),
                         seed=int(config.get("seed", 13)))


def iter_queries(bundles: Iterable[UserBundle]) -> Iterable[Tuple[UserBundle, Query]]:
    for bundle in bundles:
        for query in bundle.queries:
            yield bundle, query
