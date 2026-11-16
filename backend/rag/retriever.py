from typing import List, Dict, Any, Optional
from backend.config import config
from backend.rag.vector_store import vector_store

class RAGRetriever:
    """
    RAG Retrieval Service.
    Retrieves visual domain terminology and prompt enhancing concepts based on
    VLM structured descriptions and user prompt cues.
    RAG strictly does NOT generate images or perform forensic classification.
    """
    def __init__(self):
        self.default_top_k = config.rag.get("top_k", 5)
        self.threshold = config.rag.get("similarity_threshold", 0.1)

    def retrieve_context(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        filter_category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        k = top_k if top_k is not None else self.default_top_k
        results = vector_store.search(query=query_text, top_k=k, threshold=self.threshold)

        if filter_category:
            results = [r for r in results if r["category"].lower() == filter_category.lower()]

        return results

    def build_context_summary(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Formats retrieved documents into clean, structured context for the prompt builder."""
        if not retrieved_docs:
            return "No domain photography or lighting context retrieved."

        lines = []
        for i, doc in enumerate(retrieved_docs, start=1):
            lines.append(f"[{i}] ({doc['category'].upper()} - {doc['title']}): {doc['description']}")
        return "\n".join(lines)

rag_retriever = RAGRetriever()
