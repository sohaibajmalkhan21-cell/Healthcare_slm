"""
Embedding generation for the RAG knowledge base.

Uses a pretrained Sentence-Transformers model to convert text documents
and queries into dense vectors for semantic similarity search.
"""

from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """Wraps a Sentence-Transformers model for consistent embedding generation."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns array of shape (len(texts), embedding_dim)."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns array of shape (1, embedding_dim)."""
        return self.embed_texts([query])
