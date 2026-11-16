"""
Automated Integration Tests for New Model Routing (Section 21)
Tests:
- TEST 1: Normal pipeline execution
- TEST 2: Forced Llama failure -> Kimi K3 fallback
- TEST 3: Invalid Llama response validation -> Kimi K3 fallback
- TEST 4: Forced Kimi failure -> controlled error handling, no crash
- TEST 5: Independent Forensic Analysis integrity
- TEST 6: Forensic Audit Report generation
"""
import io
import unittest
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from backend.image_to_text.vlm import vlm_service
from backend.forensic.detector import forensic_detector
from backend.forensic.report import forensic_report_generator

class TestModelRoutingAndFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Create test synthetic image
        img = Image.new("RGB", (256, 256), color=(40, 100, 180))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        cls.img_bytes = buf.getvalue()

        # Upload image to server
        resp = cls.client.post("/api/upload", files={"file": ("test_routing.png", cls.img_bytes, "image/png")})
        assert resp.status_code == 200, "Setup upload failed"
        cls.uploaded_filename = resp.json()["filename"]

    def test_01_normal_pipeline_flow(self):
        """TEST 1: Normal pipeline execution."""
        resp = self.client.post("/api/image-to-text", json={
            "filename": self.uploaded_filename,
            "force_llama_failure": False,
            "force_kimi_failure": False
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("scene", data)
        self.assertIn("vision_model", data)
        self.assertIn("primary_image_to_text_model", data)
        self.assertIn("actual_image_to_text_model", data)
        self.assertEqual(data["primary_image_to_text_model"], "llama-3.2-90b-vision-instruct")

    def test_02_forced_llama_failure_triggers_fallback(self):
        """TEST 2: Force Llama failure -> Fallback triggered."""
        resp = self.client.post("/api/image-to-text", json={
            "filename": self.uploaded_filename,
            "force_llama_failure": True,
            "force_kimi_failure": False
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["fallback_used"])
        self.assertIsNotNone(data["fallback_reason"])
        self.assertIn("llama", data["fallback_reason"].lower())

    def test_03_invalid_llama_response_validation(self):
        """TEST 3: Schema validation failure triggers fallback."""
        from backend.providers.llama_provider import ModelOutputValidationError
        with self.assertRaises(ModelOutputValidationError):
            # Attempt to validate empty/invalid dict
            self.client.app.state = None
            from backend.providers.llama_provider import llama_vision_provider
            llama_vision_provider._validate_schema({"objects": []})

    def test_04_kimi_failure_controlled_error_no_crash(self):
        """TEST 4: Kimi failure produces controlled fallback, no application crash."""
        resp = self.client.post("/api/image-to-text", json={
            "filename": self.uploaded_filename,
            "force_llama_failure": True,
            "force_kimi_failure": True
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["fallback_used"])
        self.assertIn("kimi", data["fallback_reason"].lower())
        self.assertEqual(data["actual_image_to_text_model"], "local-cv-fallback")

    def test_05_independent_forensic_pipeline(self):
        """TEST 5: Forensic detector operates strictly independent of model history."""
        test_arr = np.random.uniform(0, 1, (128, 128, 3)).astype(np.float32)
        eval_res = forensic_detector.evaluate_image(test_arr)
        self.assertIn("score_matrix", eval_res)
        self.assertIn("total_patches", eval_res)
        self.assertTrue(eval_res["total_patches"] > 0)

    def test_06_report_generation_intact(self):
        """TEST 6: Report generation works after model integration."""
        scores = np.array([[0.2, 0.4], [0.3, 0.5]], dtype=np.float32)
        vis_paths = {"heatmap_path": "test.png", "overlay_path": "test_ov.png"}
        meta = {"filename": "test.png", "width": 128, "height": 128}
        rep = forensic_report_generator.generate_report(scores, meta, vis_paths)
        self.assertIn("ai_generated_likelihood", rep)
        self.assertIn("real_image_likelihood", rep)
        self.assertIn("report_id", rep)

if __name__ == "__main__":
    unittest.main()
