import sys
import unittest
from pathlib import Path
from PIL import Image
import json

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.image_to_text.object_detector import object_detector

class TestObjectDetection(unittest.TestCase):
    def test_detect_objects_cv(self):
        img = Image.new("RGB", (200, 200), color=(100, 150, 200))
        objs, annotated = object_detector.detect_objects_cv(img)
        self.assertIsInstance(objs, list)
        self.assertTrue(len(objs) > 0)
        self.assertIn("object", objs[0])
        self.assertIn("confidence", objs[0])
        self.assertIsNotNone(annotated)

if __name__ == "__main__":
    unittest.main()
