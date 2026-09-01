"""The per-user personal long-term memory store.

Two objects live here.

`MemoryStore` is the thing retrieval actually searches: a dense index over one
user's memories.

`MemorySketch` is the object that makes this project different from ordinary
RAG. It is a tiny, precomputed summary of the store (cluster centroids, radii,
occupancy) that can be consulted in microseconds, without touching the index.
The confidence gate reads the sketch, not the store, which is what lets the
decision happen *before* retrieval is paid for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .data import MemoryItem


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------

class TextEncoder:
    """Thin wrapper over a sentence-transformers bi-encoder.

    The model is loaded lazily on first use so that importing this repo, or
    running the unit-style checks, never triggers a download.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 64, normalize: bool = True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


# --------------------------------------------------------------------------
# Sketch
# --------------------------------------------------------------------------

@dataclass
class MemorySketch:
    """A cheap, query-independent summary of one user's memory store.

    Fields:
        centroids     (c, d) cluster centres of the stored memory vectors
        radii         (c,)   mean cosine distance from each centroid to its members
        occupancy     (c,)   fraction of memories in each cluster
        num_memories  size of the store
        mean_pairwise average cosine similarity between memories, a rough
                      measure of how topically narrow this user's store is

    Consulting the sketch costs one small matrix multiply against `centroids`,
    which is orders of magnitude cheaper than searching the full index and is
    what makes the gate a genuinely pre-retrieval decision.
    """

    centroids: np.ndarray
    radii: np.ndarray
    occupancy: np.ndarray
    num_memories: int
    mean_pairwise: float

    def features(self, query_vector: np.ndarray) -> np.ndarray:
        """Score a query against the sketch without touching the index.

        Returns eight numbers describing how well this query lands inside the
        region of embedding space the user's store actually covers.
        """
        if self.centroids.shape[0] == 0:
            return np.zeros(8, dtype=np.float32)

        similarities = self.centroids @ query_vector
        order = np.argsort(-similarities)
        best = float(similarities[order[0]])
        second = float(similarities[order[1]]) if similarities.size > 1 else best
        weighted = float(np.dot(self.occupancy, similarities))

        best_cluster = int(order[0])
        # Positive when the query sits inside the cluster's typical spread.
        inside = float(best - (1.0 - self.radii[best_cluster]))

        return np.array([
            best,
            second,
            best - second,
            weighted,
            float(np.mean(similarities)),
            float(np.std(similarities)),
            inside,
            float(np.log1p(self.num_memories)),
        ], dtype=np.float32)


def build_sketch(vectors: np.ndarray, num_clusters: int = 8,
                 seed: int = 13) -> MemorySketch:
    """Cluster a user's memory vectors into a compact sketch."""
    if vectors.shape[0] == 0:
        return MemorySketch(np.zeros((0, 1), dtype=np.float32),
                            np.zeros(0, dtype=np.float32),
                            np.zeros(0, dtype=np.float32), 0, 0.0)

    effective_clusters = int(min(num_clusters, vectors.shape[0]))
    if effective_clusters <= 1:
        centroid = vectors.mean(axis=0, keepdims=True)
        centroid /= (np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-9)
        radius = float(np.mean(vectors @ centroid[0]))
        return MemorySketch(centroid.astype(np.float32),
                            np.array([1.0 - radius], dtype=np.float32),
                            np.array([1.0], dtype=np.float32),
                            int(vectors.shape[0]),
                            _mean_pairwise(vectors))

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=effective_clusters, n_init=4, random_state=seed)
    labels = kmeans.fit_predict(vectors)
    centroids = kmeans.cluster_centers_.astype(np.float32)
    centroids /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)

    radii = np.zeros(effective_clusters, dtype=np.float32)
    occupancy = np.zeros(effective_clusters, dtype=np.float32)
    for cluster in range(effective_clusters):
        members = vectors[labels == cluster]
        occupancy[cluster] = len(members) / len(vectors)
        if len(members) > 0:
            radii[cluster] = float(1.0 - np.mean(members @ centroids[cluster]))

    return MemorySketch(centroids, radii, occupancy,
                        int(vectors.shape[0]), _mean_pairwise(vectors))


def _mean_pairwise(vectors: np.ndarray, sample: int = 256, seed: int = 13) -> float:
    """Average off-diagonal cosine similarity, subsampled for large stores."""
    if vectors.shape[0] < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    if vectors.shape[0] > sample:
        idx = rng.choice(vectors.shape[0], size=sample, replace=False)
        vectors = vectors[idx]
    gram = vectors @ vectors.T
    n = gram.shape[0]
    off_diagonal = (gram.sum() - np.trace(gram)) / (n * (n - 1))
    return float(off_diagonal)


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------

@dataclass
class RetrievedMemory:
    item: MemoryItem
    score: float


class MemoryStore:
    """A dense index over one user's memories, plus its sketch.

    The store is built once per user. `search` is the expensive path the gate
    is trying to avoid; `sketch` is the cheap path the gate is allowed to use.
    """

    def __init__(self, user_id: str, items: Sequence[MemoryItem],
                 encoder: TextEncoder, sketch_clusters: int = 8, seed: int = 13):
        self.user_id = user_id
        self.items: List[MemoryItem] = list(items)
        self.encoder = encoder
        self.vectors = encoder.encode([item.text for item in self.items])
        self.sketch = build_sketch(self.vectors, sketch_clusters, seed)
        self._index = None

    def __len__(self) -> int:
        return len(self.items)

    @property
    def index(self):
        """A FAISS inner-product index, built on first search."""
        if self._index is None and len(self.items) > 0:
            import faiss

            index = faiss.IndexFlatIP(self.vectors.shape[1])
            index.add(self.vectors)
            self._index = index
        return self._index

    def search(self, query_vector: np.ndarray, top_k: int = 4
               ) -> List[RetrievedMemory]:
        """Retrieve the top-k memories. This is the call the gate may skip."""
        if len(self.items) == 0:
            return []
        k = int(min(top_k, len(self.items)))
        index = self.index
        scores, indices = index.search(query_vector.reshape(1, -1).astype(np.float32), k)
        return [RetrievedMemory(self.items[int(i)], float(s))
                for s, i in zip(scores[0], indices[0]) if i >= 0]

    def add(self, item: MemoryItem, refresh_sketch: bool = False) -> None:
        """Append a new memory, as the memory writer does after each turn."""
        vector = self.encoder.encode([item.text])
        self.items.append(item)
        self.vectors = np.vstack([self.vectors, vector]) if len(self.vectors) else vector
        self._index = None
        if refresh_sketch:
            self.sketch = build_sketch(self.vectors, len(self.sketch.occupancy) or 8)


def build_stores(bundles, encoder: TextEncoder, sketch_clusters: int = 8,
                 seed: int = 13) -> dict:
    """Build one MemoryStore per user, keyed by user id."""
    return {
        bundle.user_id: MemoryStore(bundle.user_id, bundle.memories, encoder,
                                    sketch_clusters, seed)
        for bundle in bundles
    }
