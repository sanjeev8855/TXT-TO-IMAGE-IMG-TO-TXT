import unittest
from backend.image_to_text.vlm import VLMStructuredOutput
from backend.rag.prompt_builder import prompt_builder, PromptBoundaryValidator

class TestPromptBuilder(unittest.TestCase):
    def test_prompt_generation_and_sections(self):
        vlm_desc = VLMStructuredOutput(
            subjects=["elderly fisherman with weathered face"],
            objects=["wooden fishing boat", "net"],
            scene="harbor dock",
            environment="coastal dock with foggy ocean in background",
            lighting="soft diffuse morning light",
            composition="tight rule of thirds portrait",
            colors=["slate blue", "weathered amber"],
            style="documentary photography"
        )
        rag_docs = [
            {"category": "lighting", "title": "Golden Hour Ambient Sunlight", "description": "warm low sun"},
            {"category": "camera_optics", "title": "85mm Telephoto Portrait Lens", "description": "compression"}
        ]
        result = prompt_builder.build_prompt(vlm_desc, rag_docs, "Make it photorealistic")

        self.assertTrue(result.is_valid)
        self.assertIn("fisherman", result.sanitized_prompt.lower())
        self.assertIn("subject", result.structured_sections)
        self.assertIn("camera_view", result.structured_sections)
        self.assertIn("quality", result.structured_sections)

    def test_prompt_boundary_sanitizes_injection(self):
        validator = PromptBoundaryValidator()
        malicious_input = {
            "subject": "A person standing <script>alert(1)</script> os.system('rm -rf /') API_KEY=xyz123",
            "environment": "beach DROP TABLE users; SELECT * FROM credentials",
            "style": "cinematic photorealism"
        }
        res = validator.validate_and_sanitize(malicious_input)
        self.assertTrue(res.sanitization_applied)
        self.assertTrue(len(res.violations_detected) > 0)
        self.assertNotIn("<script>", res.sanitized_prompt)
        self.assertNotIn("DROP TABLE", res.sanitized_prompt)
        self.assertNotIn("os.system", res.sanitized_prompt)

if __name__ == "__main__":
    unittest.main()
