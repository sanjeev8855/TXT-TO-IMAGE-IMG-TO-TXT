import io
import unittest
from PIL import Image
from backend.image_to_text.preprocessing import validate_image_bytes, preprocess_image, ImagePreprocessingError

class TestImagePreprocessing(unittest.TestCase):
    def setUp(self):
        # Create a valid test image in memory
        img = Image.new("RGB", (256, 256), color=(128, 64, 32))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.valid_png_bytes = buf.getvalue()

    def test_valid_image(self):
        meta = validate_image_bytes(self.valid_png_bytes, "test.png")
        self.assertTrue(meta["valid"])
        self.assertEqual(meta["width"], 256)
        self.assertEqual(meta["height"], 256)

        rgb_img, np_arr, p_meta = preprocess_image(self.valid_png_bytes)
        self.assertEqual(rgb_img.size, (256, 256))
        self.assertEqual(np_arr.shape, (256, 256, 3))
        self.assertAlmostEqual(np_arr.min(), 32 / 255.0, places=2)
        self.assertAlmostEqual(np_arr.max(), 128 / 255.0, places=2)

    def test_empty_bytes(self):
        with self.assertRaises(ImagePreprocessingError):
            validate_image_bytes(b"", "empty.png")

    def test_too_small_dimensions(self):
        tiny_img = Image.new("RGB", (32, 32), color="red")
        buf = io.BytesIO()
        tiny_img.save(buf, format="PNG")
        with self.assertRaises(ImagePreprocessingError):
            validate_image_bytes(buf.getvalue(), "tiny.png")

    def test_unsupported_format(self):
        with self.assertRaises(ImagePreprocessingError):
            validate_image_bytes(b"not an image text content", "test.txt")

if __name__ == "__main__":
    unittest.main()
