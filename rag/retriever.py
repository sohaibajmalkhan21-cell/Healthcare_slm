"""
High-level retriever: ties embedding model + vector store together.
"""

from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore, load_knowledge_base


class MedicalRetriever:
    def __init__(self, knowledge_base_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = EmbeddingModel(model_name)
        self.documents = load_knowledge_base(knowledge_base_path)

        self.store = VectorStore(embedding_dim=self.embedding_model.embedding_dim)

        texts = [f"{doc['title']}. {doc['content']}" for doc in self.documents]
        embeddings = self.embedding_model.embed_texts(texts)
        self.store.add_documents(self.documents, embeddings)

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        query_embedding = self.embedding_model.embed_query(query)
        return self.store.search(query_embedding, k=k)


if __name__ == "__main__":
    retriever = MedicalRetriever("data/knowledge_base/vitals_reference.json")

    test_queries = [
        "patient has elevated pulse and low oxygen",
        "what is a normal body temperature",
        "patient is breathing very fast",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, k=2)
        for r in results:
            print(f"  [{r['similarity_score']:.3f}] {r['title']}")
