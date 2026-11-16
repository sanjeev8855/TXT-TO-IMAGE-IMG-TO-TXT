import io
import unittest
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app

class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Create a test image
        img = Image.new("RGB", (256, 256), color=(70, 130, 180))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.test_img_bytes = buf.getvalue()

    def test_health_check(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "healthy")

    def test_full_pipeline_flow(self):
        # 1. Upload
        files = {"file": ("test_sky.png", self.test_img_bytes, "image/png")}
        up_resp = self.client.post("/api/upload", files=files)
        self.assertEqual(up_resp.status_code, 200)
        up_data = up_resp.json()
        filename = up_data["filename"]
        self.assertTrue(len(filename) > 0)

        # 2. VLM
        vlm_resp = self.client.post("/api/image-to-text", json={"filename": filename})
        self.assertEqual(vlm_resp.status_code, 200)
        vlm_data = vlm_resp.json()
        self.assertIn("scene", vlm_data)
        self.assertIn("lighting", vlm_data)

        # 3. RAG Retrieval
        rag_resp = self.client.post("/api/rag-retrieve", json={"query": "coastal sunset natural photography", "top_k": 3})
        self.assertEqual(rag_resp.status_code, 200)
        rag_data = rag_resp.json()
        self.assertEqual(len(rag_data["retrieved_documents"]), 3)

        # 4. Prompt Construction & Boundary Check
        p_resp = self.client.post("/api/prompt-build", json={
            "vlm_output": vlm_data,
            "retrieved_documents": rag_data["retrieved_documents"],
            "user_instructions": "cinematic atmosphere"
        })
        self.assertEqual(p_resp.status_code, 200)
        prompt_data = p_resp.json()
        self.assertTrue(prompt_data["is_valid"])
        sanitized_prompt = prompt_data["sanitized_prompt"]
        self.assertTrue(len(sanitized_prompt) > 10)

        # 5. Text-to-Image Generation
        gen_resp = self.client.post("/api/text-to-image", json={
            "prompt": sanitized_prompt,
            "seed": 42,
            "steps": 10
        })
        self.assertEqual(gen_resp.status_code, 200)
        gen_data = gen_resp.json()
        gen_filename = gen_data["image_filename"]

        # 6. Forensic AI Detection & Heatmap
        forensic_resp = self.client.post("/api/forensics/analyze", json={
            "filename": gen_filename,
            "alpha": 0.45
        })
        self.assertEqual(forensic_resp.status_code, 200)
        forensic_data = forensic_resp.json()
        report = forensic_data["report"]
        self.assertIn("ai_generated_likelihood", report)
        self.assertIn("real_image_likelihood", report)
        self.assertIn("visualizations", report)
        self.assertIn("heatmap_url", report["visualizations"])
        self.assertIn("overlay_url", report["visualizations"])

        # 7. Experiment Logger check
        exp_resp = self.client.get("/api/experiments?limit=5")
        self.assertEqual(exp_resp.status_code, 200)
        exps = exp_resp.json()
        self.assertTrue(len(exps) > 0)

if __name__ == "__main__":
    unittest.main()
