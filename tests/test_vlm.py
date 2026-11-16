import unittest
from PIL import Image
from backend.image_to_text.vlm import vlm_service, VLMStructuredOutput

class TestVLM(unittest.TestCase):
    def test_local_vlm_output_structure(self):
        img = Image.new("RGB", (400, 300), color=(200, 100, 50))
        output = vlm_service.analyze_image(img)
        self.assertIsInstance(output, VLMStructuredOutput)
        self.assertIsInstance(output.subjects, list)
        self.assertIsInstance(output.objects, list)
        self.assertIsInstance(output.colors, list)
        self.assertTrue(len(output.scene) > 0)
        self.assertTrue(len(output.composition) > 0)
        self.assertIn("Warm red/amber spectrum", output.colors)
        self.assertIn("direct_observations", output.observations_vs_assumptions)
        self.assertIn("contextual_assumptions", output.observations_vs_assumptions)

if __name__ == "__main__":
    unittest.main()
