import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from backend.config import config
from backend.rag.embeddings import embedding_engine

class RAGDocument:
    def __init__(self, doc_id: str, category: str, title: str, description: str, keywords: List[str]):
        self.doc_id = doc_id
        self.category = category
        self.title = title
        self.description = description
        self.keywords = keywords
        self.full_text = f"{title}. {description}. Keywords: {', '.join(keywords)}"
        self.embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.doc_id,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "keywords": self.keywords,
        }

class LocalVectorStore:
    def __init__(self, docs_dir: Optional[Path] = None):
        self.docs_dir = docs_dir or config.rag_docs_dir
        self.documents: List[RAGDocument] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self.load_documents()

    def load_documents(self):
        """Loads all JSON documents from the RAG knowledge directory and indexes them."""
        self.documents = []
        if not self.docs_dir.exists():
            print(f"[RAG] Warning: Documents directory {self.docs_dir} does not exist.")
            return

        json_files = list(self.docs_dir.glob("*.json"))
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    items = json.load(f)
                    if isinstance(items, list):
                        for item in items:
                            doc = RAGDocument(
                                doc_id=item.get("id", f"doc_{len(self.documents)+1}"),
                                category=item.get("category", jf.stem),
                                title=item.get("title", ""),
                                description=item.get("description", ""),
                                keywords=item.get("keywords", [])
                            )
                            self.documents.append(doc)
            except Exception as e:
                print(f"[RAG] Error reading {jf}: {e}")

        # Compute embeddings for all documents
        if self.documents:
            vectors = []
            for doc in self.documents:
                doc.embedding = embedding_engine.get_embedding(doc.full_text)
                vectors.append(doc.embedding)
            self.doc_embeddings = np.vstack(vectors)
        else:
            self.doc_embeddings = None

        print(f"[RAG] Loaded and indexed {len(self.documents)} domain knowledge documents.")

    def search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Executes cosine similarity search between query embedding and all knowledge vectors.
        Returns top_k matching documents with scores.
        """
        if not self.documents or self.doc_embeddings is None:
            return []

        query_vec = embedding_engine.get_embedding(query)
        # Cosine similarity: (N, D) @ (D,)
        scores = np.dot(self.doc_embeddings, query_vec)

        # Rank indices by descending score
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices[:top_k]:
            score = float(scores[idx])
            doc = self.documents[idx]
            results.append({
                "id": doc.doc_id,
                "category": doc.category,
                "title": doc.title,
                "description": doc.description,
                "keywords": doc.keywords,
                "similarity_score": round(score, 4),
                "retrieval_justification": f"Ranked #{len(results)+1} match in '{doc.category}' with cosine similarity {score:.3f}"
            })

        return results

vector_store = LocalVectorStore()
