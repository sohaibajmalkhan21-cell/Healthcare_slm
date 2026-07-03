"""
FAISS-backed vector store for the medical knowledge base.
"""

import json
import faiss
import numpy as np


class VectorStore:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.documents: list[dict] = []

    def add_documents(self, documents: list[dict], embeddings: np.ndarray):
        normalized = self._normalize(embeddings)
        self.index.add(normalized)
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, k: int = 3) -> list[dict]:
        normalized_query = self._normalize(query_embedding)
        scores, indices = self.index.search(normalized_query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = dict(self.documents[idx])
            doc["similarity_score"] = float(score)
            results.append(doc)
        return results

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return vectors / norms


def load_knowledge_base(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
