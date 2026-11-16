import sys
import unittest
from pathlib import Path
from PIL import Image
import json

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.image_to_text.color_extractor import cv_color_extractor

class TestColorExtraction(unittest.TestCase):
    def test_palette_extraction(self):
        img = Image.new("RGB", (200, 200), color=(240, 160, 90))
        res = cv_color_extractor.extract_palette(img)
        self.assertIn("dominant_colors", res)
        self.assertIn("color_groups", res)
        self.assertTrue(len(res["dominant_colors"]) > 0)
        self.assertIn("hex", res["dominant_colors"][0])

if __name__ == "__main__":
    unittest.main()
