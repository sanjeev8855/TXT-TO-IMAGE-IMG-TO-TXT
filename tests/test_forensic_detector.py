import unittest
from PIL import Image
import numpy as np

from backend.forensic.patch_extractor import patch_extractor
from backend.forensic.detector import forensic_detector
from backend.forensic.anomaly_map import anomaly_map_generator
from backend.forensic.heatmap import heatmap_generator
from backend.forensic.report import forensic_report_generator

class TestForensics(unittest.TestCase):
    def setUp(self):
        # Create a test synthetic-like array (256x256x3) with high frequency grid
        h, w = 256, 256
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)
        test_arr = np.zeros((h, w, 3), dtype=np.float32)
        test_arr[:, :, 0] = 0.5 + 0.3 * np.sin(xx * 32 * np.pi)
        test_arr[:, :, 1] = 0.4 + 0.2 * np.cos(yy * 32 * np.pi)
        test_arr[:, :, 2] = 0.6
        self.np_img = np.clip(test_arr, 0.0, 1.0)
        self.pil_img = Image.fromarray((self.np_img * 255).astype(np.uint8))

    def test_patch_extraction(self):
        patches, rows, cols = patch_extractor.extract_patches(self.np_img)
        self.assertTrue(len(patches) > 0)
        self.assertEqual(len(patches), rows * cols)
        self.assertEqual(patches[0].array.shape, (64, 64, 3))

    def test_forensic_evaluation(self):
        res = forensic_detector.evaluate_image(self.np_img)
        self.assertIn("score_matrix", res)
        self.assertIn("fft_spectrum_array", res)
        self.assertIn("noise_residual_array", res)
        score_mat = res["score_matrix"]
        self.assertTrue(score_mat.min() >= 0.0)
        self.assertTrue(score_mat.max() <= 1.0)

    def test_anomaly_interpolation_and_heatmap(self):
        eval_res = forensic_detector.evaluate_image(self.np_img)
        score_mat = eval_res["score_matrix"]
        cont_map = anomaly_map_generator.generate_continuous_map(score_mat, (256, 256))
        self.assertEqual(cont_map.shape, (256, 256))

        vis_paths = heatmap_generator.save_forensic_visualizations(
            self.pil_img,
            cont_map,
            fft_array=eval_res["fft_spectrum_array"],
            residual_array=eval_res["noise_residual_array"]
        )
        self.assertIn("heatmap_path", vis_paths)
        self.assertIn("overlay_path", vis_paths)
        self.assertIn("fft_path", vis_paths)
        self.assertIn("residual_path", vis_paths)

    def test_forensic_report_and_disclaimer(self):
        eval_res = forensic_detector.evaluate_image(self.np_img)
        vis_paths = {"heatmap_path": "test.png", "overlay_path": "test_ov.png"}
        report = forensic_report_generator.generate_report(
            eval_res["score_matrix"],
            {"width": 256, "height": 256},
            vis_paths
        )
        self.assertIn("ai_generated_likelihood", report)
        self.assertIn("real_image_likelihood", report)
        self.assertIn("anomaly_level", report)
        self.assertIn("academic_disclaimer", report)
        self.assertIn("model assessment", report["academic_disclaimer"].lower())
        self.assertNotIn("definitely ai-generated", report["academic_disclaimer"].lower())
        self.assertEqual(round(report["ai_generated_likelihood"] + report["real_image_likelihood"]), 100)

if __name__ == "__main__":
    unittest.main()
