import os
import unittest
from backend.text_to_image.generator import image_generator

class TestTextToImage(unittest.TestCase):
    def test_image_generation_pipeline(self):
        prompt = "A photorealistic portrait of an astronomer with telescope, warm lighting, 35mm lens"
        res = image_generator.generate_image(prompt, user_seed=123, custom_steps=10)

        self.assertIn("image_filename", res)
        self.assertIn("image_path", res)
        self.assertTrue(os.path.exists(res["image_path"]))
        self.assertEqual(res["seed"], 123)
        self.assertEqual(res["steps"], 10)
        self.assertIsInstance(res["process_timeline"], list)
        self.assertEqual(len(res["process_timeline"]), 5)

        step_names = [s["name"] for s in res["process_timeline"]]
        self.assertTrue(any("1. Encoding prompt" in name for name in step_names))
        self.assertTrue(any("3. Denoising" in name for name in step_names))
        self.assertTrue(any("5. Image generated" in name for name in step_names))

        self.assertIn("scheduler", res["technical_details"])
        self.assertIn("latent_channels", res["technical_details"])

if __name__ == "__main__":
    unittest.main()
