import unittest
from backend.rag.embeddings import embedding_engine
from backend.rag.vector_store import vector_store
from backend.rag.retriever import rag_retriever

class TestRAG(unittest.TestCase):
    def test_embedding_shape_and_norm(self):
        vec = embedding_engine.get_embedding("golden hour lighting bokeh portrait")
        self.assertEqual(len(vec), 256)
        # Check L2 norm is ~1.0
        norm = sum(x**2 for x in vec) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=2)

    def test_retrieval_returns_relevant_docs(self):
        docs = rag_retriever.retrieve_context("golden hour sunset photography portrait bokeh", top_k=3)
        self.assertTrue(len(docs) > 0)
        self.assertLessEqual(len(docs), 3)

        doc_titles = [d["title"].lower() for d in docs]
        # At least one doc should relate to lighting or photography
        categories = [d["category"] for d in docs]
        self.assertTrue(any(c in ["lighting", "photography", "camera_optics"] for c in categories))

    def test_configurable_top_k(self):
        docs_1 = rag_retriever.retrieve_context("cinematic camera", top_k=1)
        self.assertEqual(len(docs_1), 1)

        docs_4 = rag_retriever.retrieve_context("cinematic camera", top_k=4)
        self.assertLessEqual(len(docs_4), 4)

if __name__ == "__main__":
    unittest.main()
