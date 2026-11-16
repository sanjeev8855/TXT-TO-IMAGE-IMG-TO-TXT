import re
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from backend.config import config
from backend.image_to_text.vlm import VLMStructuredOutput

class StructuredPromptSchema(BaseModel):
    subject: str = Field(..., description="Primary subject(s) in the frame")
    environment: str = Field(..., description="Setting, background, and environment")
    composition: str = Field(..., description="Framing, balance, aspect, and perspective")
    lighting: str = Field(..., description="Lighting direction, quality, temperature, and mood")
    camera_view: str = Field(..., description="Camera system, focal length, aperture, or lens characteristics")
    style: str = Field(..., description="Aesthetic style (e.g., editorial photograph, cinematic still)")
    color: str = Field(..., description="Color grading, palette, and tonal balance")
    material_texture: str = Field(..., description="Physical materials, skin texture, surface fidelity")
    detail: str = Field(..., description="Fine contextual micro-details")
    quality: str = Field(..., description="Rendering quality and optical fidelity descriptors")

class PromptValidationResult(BaseModel):
    is_valid: bool
    sanitized_prompt: str
    structured_sections: Dict[str, str]
    violations_detected: List[str]
    sanitization_applied: bool

class PromptBoundaryValidator:
    """
    Prompt Boundary and Validation Layer (Module 6).
    Enforces that final prompts strictly contain image-generation relevant visual concepts.
    Rejects and strips code, SQL, shell commands, script tags, API keys, and injection attempts.
    """
    def __init__(self):
        pb_conf = config.prompt_boundary
        self.max_length = pb_conf.get("max_length", 1200)
        self.forbidden_regexes = [
            re.compile(pat, re.IGNORECASE) for pat in pb_conf.get("forbidden_patterns", [])
        ]
        # Additional system safety patterns
        self.code_patterns = [
            re.compile(r"<\s*script[^>]*>.*?<\s*/\s*script\s*>", re.IGNORECASE | re.DOTALL),
            re.compile(r"(rm\s+-rf|sudo\s+|chmod\s+|wget\s+|curl\s+|bash\b|cmd\.exe|powershell\.exe)", re.IGNORECASE),
            re.compile(r"(SELECT\s+.*?\s+FROM|UNION\s+SELECT|DROP\s+TABLE|INSERT\s+INTO)", re.IGNORECASE),
            re.compile(r"(process\.env|os\.environ|eval\(|exec\()", re.IGNORECASE),
        ]

    def validate_and_sanitize(self, raw_sections: Dict[str, str]) -> PromptValidationResult:
        violations = []
        sanitized_sections = {}
        was_sanitized = False

        for section_key, val in raw_sections.items():
            cleaned_val = str(val or "").strip()

            # Check for forbidden patterns
            for pat in self.forbidden_regexes + self.code_patterns:
                if pat.search(cleaned_val):
                    matched = pat.pattern
                    violations.append(f"Detected prohibited pattern '{matched}' in section '{section_key}'.")
                    cleaned_val = pat.sub("[SANITIZED]", cleaned_val)
                    was_sanitized = True

            # Strip html brackets
            cleaned_val = re.sub(r"[<>]", "", cleaned_val)
            sanitized_sections[section_key] = cleaned_val

        # Construct final unified prompt
        prompt_parts = []
        for key in ["style", "subject", "environment", "composition", "lighting", "camera_view", "material_texture", "color", "detail", "quality"]:
            val = sanitized_sections.get(key, "").strip()
            if val and val != "[SANITIZED]":
                prompt_parts.append(val)

        final_prompt = ", ".join(prompt_parts)

        # Enforce max length boundary
        if len(final_prompt) > self.max_length:
            violations.append(f"Prompt length ({len(final_prompt)} chars) exceeded limit ({self.max_length}). Truncating safely.")
            final_prompt = final_prompt[:self.max_length].rsplit(" ", 1)[0]
            was_sanitized = True

        is_valid = len(violations) == 0 or was_sanitized

        return PromptValidationResult(
            is_valid=is_valid,
            sanitized_prompt=final_prompt,
            structured_sections=sanitized_sections,
            violations_detected=violations,
            sanitization_applied=was_sanitized
        )

class LLMPromptBuilder:
    """
    Transforms VLM Structured Descriptions + RAG Retrieved Context + User Instructions
    into a structured 10-part image prompt and validates it through the Prompt Boundary Layer.
    """
    def __init__(self):
        self.validator = PromptBoundaryValidator()
        self.api_key = config.gemini_api_key
        self.llm_model = config.models.get("llm_model", "gemini-3.7-flash")
        if self.llm_model in ("gemini-2.5-flash", "gemini-2.0-flash"):
            self.llm_model = "gemini-3.7-flash"

    def build_prompt(
        self,
        vlm_desc: VLMStructuredOutput,
        retrieved_docs: List[Dict[str, Any]],
        user_instructions: Optional[str] = None
    ) -> PromptValidationResult:
        # 1. Attempt Kimi K3 reasoning prompt enhancement if configured
        if config.kimi_api_key and not config.force_kimi_failure:
            try:
                from backend.providers.kimi_provider import kimi_provider
                raw_sections = kimi_provider.enhance_prompt_reasoning(
                    rag_context_docs=retrieved_docs,
                    structured_desc=vlm_desc.model_dump(),
                    user_instructions=user_instructions
                )
                return self.validator.validate_and_sanitize(raw_sections)
            except Exception as e:
                print(f"[PromptBuilder] Kimi K3 prompt reasoning fallback ({e}). Falling back to primary LLM.")

        # 2. Primary LLM reasoning (Gemini)
        if self.api_key:
            try:
                raw_sections = self._generate_with_gemini_llm(vlm_desc, retrieved_docs, user_instructions)
                return self.validator.validate_and_sanitize(raw_sections)
            except Exception as e:
                print(f"[PromptBuilder] Gemini LLM synthesis failed ({e}). Using local synthesis engine.")

        # 3. Local synthesis fallback
        raw_sections = self._generate_local_synthesis(vlm_desc, retrieved_docs, user_instructions)
        return self.validator.validate_and_sanitize(raw_sections)

    def _generate_local_synthesis(
        self,
        vlm: VLMStructuredOutput,
        retrieved_docs: List[Dict[str, Any]],
        user_instructions: Optional[str] = None
    ) -> Dict[str, str]:
        """Rule-based synthesis engine combining visual entities with retrieved RAG optics & styles."""
        # Subjects & environment with heuristic stripping
        raw_subjects = vlm.subjects if vlm.subjects else ["Visual subject"]
        clean_subjects = []
        for s in raw_subjects:
            cs = s
            for junk in ["(Local Heuristic Engine)", "(Feature extracted)", "pixel substrate", "Digital Visual Asset", "Localized spatial visual clusters"]:
                cs = cs.replace(junk, "").strip()
            cs = re.sub(r"\s+", " ", cs).strip(" ,")
            if cs:
                clean_subjects.append(cs)
        subjects_str = ", ".join(clean_subjects) if clean_subjects else "Scenic natural landscape"

        env_str = vlm.environment or vlm.scene or "Natural ambient environment"
        for junk in ["(Local Heuristic Engine)", "(Feature extracted)", "pixel substrate"]:
            env_str = env_str.replace(junk, "").strip()
        env_str = re.sub(r"\s+", " ", env_str).strip(" ,")

        # Selective RAG enrichment: Pick top relevant term per category instead of concatenating contradictory tags
        chosen_lighting = None
        for doc in retrieved_docs:
            if doc.get("category") == "lighting":
                t = doc.get("title", "")
                is_cyber_doc = "cyberpunk" in t.lower() or "neon" in t.lower()
                query_context = (subjects_str + " " + env_str + " " + (user_instructions or "")).lower()
                has_cyber_intent = any(k in query_context for k in ["cyberpunk", "neon", "sci-fi", "futuristic", "night city"])
                if is_cyber_doc and not has_cyber_intent:
                    continue
                chosen_lighting = t
                break

        chosen_comp = None
        for doc in retrieved_docs:
            if doc.get("category") == "composition":
                chosen_comp = doc.get("title", "")
                break

        chosen_camera = None
        for doc in retrieved_docs:
            if doc.get("category") in ["photography", "camera_optics"]:
                chosen_camera = doc.get("title", "")
                break

        chosen_style = None
        for doc in retrieved_docs:
            if doc.get("category") == "art_styles":
                chosen_style = doc.get("title", "")
                break

        chosen_quality = None
        for doc in retrieved_docs:
            if doc.get("category") == "quality_descriptors":
                chosen_quality = doc.get("title", "")
                break

        lighting_term = chosen_lighting or vlm.lighting or "Warm natural directional lighting"
        camera_term = chosen_camera or "35mm photographic perspective with natural depth of field"

        raw_style = vlm.style or ""
        if any(k in raw_style for k in ["Awaiting", "Heuristic", "Digital Visual Asset", "Local"]):
            style_term = chosen_style or "Photorealistic photograph"
        else:
            style_term = chosen_style or raw_style or "Photorealistic photograph"

        comp_term = chosen_comp or vlm.composition or "Centered composition with balanced framing"
        quality_term = chosen_quality or "High optical fidelity, realistic micro-textures, 8k resolution"

        colors_term = ", ".join(vlm.colors) if vlm.colors else "Balanced natural color palette"
        details_term = "; ".join(vlm.visual_details) if vlm.visual_details else "Rich surface textures and authentic environmental details"

        if user_instructions:
            style_term = f"{style_term}, {user_instructions}"

        return {
            "subject": f"A detailed visual depiction of {subjects_str}",
            "environment": env_str,
            "composition": comp_term,
            "lighting": lighting_term,
            "camera_view": camera_term,
            "style": style_term,
            "color": colors_term,
            "material_texture": "Authentic physical materials and tactile micro-textures",
            "detail": details_term,
            "quality": quality_term
        }

    def _generate_with_gemini_llm(
        self,
        vlm: VLMStructuredOutput,
        retrieved_docs: List[Dict[str, Any]],
        user_instructions: Optional[str]
    ) -> Dict[str, str]:
        import requests

        rag_text = "\n".join([f"- {d['category'].upper()}: {d['title']} - {d['description']}" for d in retrieved_docs])
        prompt_input = f"""You are a professional prompt engineer for advanced image synthesis.
Transform the following VLM image breakdown, retrieved photography/visual knowledge, and user instructions into a structured 10-part image prompt.

VLM Visual Description:
- Subjects: {', '.join(vlm.subjects)}
- Environment: {vlm.environment}
- Scene: {vlm.scene}
- Lighting: {vlm.lighting}
- Composition: {vlm.composition}
- Colors: {', '.join(vlm.colors)}

Retrieved Domain Knowledge (RAG):
{rag_text}

Optional User Request:
{user_instructions or 'None specified.'}

Return JSON with exactly these 10 keys:
"subject", "environment", "composition", "lighting", "camera_view", "style", "color", "material_texture", "detail", "quality".
Each value must be a concise, highly descriptive visual phrase.
Do NOT output markdown ticks or code explanations. Return only valid JSON.
"""
        payload = {
            "contents": [{"parts": [{"text": prompt_input}]}],
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.4}
        }

        # Try multiple Gemini models in order — skip on 429/503 quota issues
        llm_models_to_try = [
            "gemini-flash-lite-latest",  # confirmed working ultra-fast
            "gemini-3.1-flash-lite",     # confirmed working
            "gemini-3-flash-preview",    # confirmed working
            "gemini-3.8-flash",          # latest official model
            self.llm_model,
        ]
        seen_m: set = set()
        llm_models_to_try = [m for m in llm_models_to_try if not (m in seen_m or seen_m.add(m))]

        last_err = None
        for model_name in llm_models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
            try:
                resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=4)
                if resp.status_code == 200:
                    res_json = resp.json()
                    raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                    return json.loads(raw_text)
                elif resp.status_code in (429, 503):
                    print(f"[PromptBuilder] Gemini {model_name} returned {resp.status_code}, trying next model.")
                    continue
                else:
                    last_err = f"HTTP {resp.status_code} from {model_name}"
                    print(f"[PromptBuilder] Gemini {model_name} returned {resp.status_code}, trying next.")
                    continue
            except Exception as e:
                last_err = f"{model_name}: {type(e).__name__}"
                print(f"[PromptBuilder] Gemini {model_name} failed: {type(e).__name__}")
                continue

        raise RuntimeError(f"All Gemini LLM models failed. Last: {last_err}")


prompt_builder = LLMPromptBuilder()
