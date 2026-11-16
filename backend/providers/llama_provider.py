"""
Llama 3.2 90B Vision Instruct Provider (Primary Image -> Text Model)
Handles multimodal inference with Gemini visual context injection and strict schema validation.
"""
import io
import re
import json
import time
import base64
from typing import Dict, Any, List, Optional
from PIL import Image
import requests

from backend.config import config

class ModelOutputValidationError(Exception):
    """Raised when an Image-to-Text model returns an invalid, empty, or unparseable response."""
    pass

def _format_list(val: Any) -> str:
    if isinstance(val, list):
        items = [str(x).strip() for x in val if x is not None and str(x).strip()]
        return ", ".join(items) if items else "N/A"
    if isinstance(val, str) and val.strip():
        return val.strip()
    return "N/A"

class LlamaVisionProvider:
    """
    Primary Image -> Text provider using Llama 3.2 90B Vision Instruct.
    Compatible with Groq, Together AI, OpenRouter, Hugging Face, or OpenAI-compatible vLLM endpoints.
    """
    def __init__(self):
        self.api_key = config.llama_api_key
        self.model = config.llama_model or "llama-3.2-90b-vision-instruct"
        self.api_base = config.llama_api_base

    @property
    def provider_name(self) -> str:
        return "Llama 3.2 90B Vision Instruct"

    def _resolve_endpoint(self) -> str:
        if self.api_base:
            base = self.api_base.rstrip("/")
            return f"{base}/chat/completions" if not base.endswith("/chat/completions") else base

        key = self.api_key.strip()
        if key.startswith("gsk_"):
            return "https://api.groq.com/openai/v1/chat/completions"
        elif key.startswith("sk-or-"):
            return "https://openrouter.ai/api/v1/chat/completions"
        elif key.startswith("hf_"):
            return "https://api-inference.huggingface.co/v1/chat/completions"
        else:
            return "https://api.together.xyz/v1/chat/completions"

    def generate_structured_description(
        self,
        pil_image: Image.Image,
        gemini_context: Optional[Dict[str, Any]] = None,
        cv_color_data: Optional[Dict[str, Any]] = None,
        force_fail: bool = False
    ) -> Dict[str, Any]:
        """
        Executes Llama 3.2 90B Vision Instruct inference and validates output.
        Raises ModelOutputValidationError on any failure so the caller can trigger Kimi K3.
        """
        if force_fail or config.force_llama_failure:
            raise ModelOutputValidationError("Simulated Llama 3.2 90B Vision failure for fallback evaluation.")

        if not self.api_key:
            raise ModelOutputValidationError("LLAMA_API_KEY is not configured on the server.")

        # Encode image to optimized lightweight base64 buffer for rapid NIM transport
        thumb = pil_image.copy()
        thumb.thumbnail((512, 512))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=82)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Synthesize context prompt including Gemini visual observations & CV color data
        gemini_summary = ""
        if gemini_context:
            gemini_summary = f"""
OBSERVATIONS FROM GEMINI VISION STAGE:
- Scene Category: {gemini_context.get('scene', 'N/A')}
- Primary Subjects: {_format_list(gemini_context.get('subjects'))}
- Lighting Characteristics: {gemini_context.get('lighting', 'N/A')}
- Composition & Framing: {gemini_context.get('composition', 'N/A')}
- Environment / Background: {gemini_context.get('environment', 'N/A')}
- Direct Visual Facts: {gemini_context.get('observations_vs_assumptions', {}).get('direct_observations', 'N/A')}
"""

        cv_summary = ""
        if cv_color_data:
            raw_cols = cv_color_data.get("dominant_colors", [])[:5]
            col_strs = []
            for c in raw_cols:
                if isinstance(c, dict):
                    name = c.get("name", "Color")
                    h = c.get("hex", "")
                    pct = c.get("percentage", "")
                    col_strs.append(f"{name} ({h}, {pct}%)" if h else name)
                elif isinstance(c, str):
                    col_strs.append(c)
            if col_strs:
                cv_summary = f"\nVERIFIED COMPUTER-VISION PIXEL COLORS:\n{', '.join(col_strs)}\n"

        prompt_instruction = f"""You are Llama 3.2 90B Vision Instruct, an expert multimodal computer vision researcher.
Analyze this image and synthesize a comprehensive structured image description.
{gemini_summary}
{cv_summary}
You MUST respond with a single valid JSON object adhering to this schema:
{{
  "scene": "Comprehensive scene categorization",
  "summary": "Concise high-level visual summary",
  "detailed_description": "Exhaustive description of entities, visual textures, and environment",
  "subjects": ["List of primary subjects"],
  "objects": ["List of key visible objects"],
  "object_descriptions": [
    {{
      "name": "Object name",
      "location": "Center / Left / Right / Foreground",
      "dominant_color": "Dominant color",
      "material": "Material if visible or 'Uncertain / low confidence'"
    }}
  ],
  "foreground": "Detailed foreground description",
  "background": "Detailed background description",
  "environment": "Indoor / outdoor setting",
  "composition": "Framing, symmetry, perspective, rule-of-thirds",
  "lighting": "Direction, quality, color temperature, and shadow profile",
  "atmosphere": "Atmospheric mood and tone",
  "style": "Photographic realism / Digital art / Editorial / etc.",
  "colors": ["List of observed dominant colors"],
  "semantic_colors": ["Warm amber", "Deep cyan", "Natural green"],
  "visual_details": ["Key tactile textures, materials, and fine details"],
  "possible_camera_characteristics": ["35mm lens equivalent", "f/2.8 aperture"],
  "generation_prompt": "A generation-ready natural language text prompt to reproduce this exact image"
}}
Do NOT output markdown commentary or conversational text. Return strictly raw valid JSON.
"""

        endpoint = self._resolve_endpoint()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Llama 3.2 90B Vision Instruct on NIM
        # Note: 11B reached EOL on NIM (2026-11-05), so we query 90B directly
        models_to_try = [self.model]

        last_err = None
        for m_name in models_to_try:
            payload = {
                "model": m_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are Llama 3.2 Vision Instruct, an expert multimodal computer vision researcher. "
                            "You MUST analyze the image and respond with a single valid JSON object adhering to the schema. "
                            "Output strictly valid raw JSON without conversational preamble or markdown code blocks."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_instruction},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        ]
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 1000
            }

            # 8s timeout: if NIM queue is clear, responds; if congested, quickly failover to fallback provider
            t_out = 8
            try:
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=t_out)
                if resp.status_code == 401:
                    raise ModelOutputValidationError(f"Authentication failed for {m_name}: invalid API key")
                if resp.status_code == 400:
                    raise ModelOutputValidationError(f"Bad request to {m_name}: {resp.text[:200]}")
                if resp.status_code == 422:
                    raise ModelOutputValidationError(f"Model {m_name} rejected request (422): {resp.text[:200]}")
                if resp.status_code == 429:
                    raise ModelOutputValidationError(f"Rate limit exceeded for {m_name} (429)")
                if resp.status_code != 200:
                    raise ModelOutputValidationError(f"HTTP {resp.status_code} from {m_name}: {resp.text[:200]}")

                res_json = resp.json()
                choices = res_json.get("choices", [])
                if not choices:
                    raise ModelOutputValidationError(f"Empty choices from {m_name}")

                raw_content = choices[0]["message"]["content"].strip()
                if not raw_content:
                    raise ModelOutputValidationError(f"Empty content from {m_name}")

                # Robust JSON extraction handling markdown fences or preamble
                clean_json_str = raw_content
                fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_content, re.DOTALL)
                if fence_match:
                    clean_json_str = fence_match.group(1).strip()
                else:
                    first_b = raw_content.find("{")
                    last_b = raw_content.rfind("}")
                    if first_b != -1 and last_b != -1 and last_b > first_b:
                        clean_json_str = raw_content[first_b : last_b + 1].strip()

                parsed = None
                try:
                    parsed = json.loads(clean_json_str)
                except Exception:
                    json_match = re.search(r"\{[\s\S]*\}", raw_content)
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group(0))
                        except Exception:
                            parsed = None

                # If Llama returned natural prose rather than strict JSON, wrap its rich description:
                if parsed is None or not isinstance(parsed, dict):
                    first_line = raw_content.strip().split("\n")[0]
                    subj = (gemini_context.get("subjects") if gemini_context else None) or ["Observed visual subject"]
                    obj = (gemini_context.get("objects") if gemini_context else None) or ["Observed scene entity"]
                    cols = (cv_color_data.get("dominant_colors") if cv_color_data else None) or ["Natural palette"]
                    parsed = {
                        "scene": first_line[:100] if len(first_line) > 5 else "Observed visual scene",
                        "summary": raw_content[:300].strip(),
                        "detailed_description": raw_content.strip(),
                        "subjects": subj,
                        "objects": obj,
                        "foreground": "Foreground visual elements",
                        "background": "Ambient background context",
                        "environment": "Natural ambient environment",
                        "composition": "Centered balanced composition",
                        "lighting": "Natural ambient lighting",
                        "atmosphere": "Atmospheric mood",
                        "style": "Photographic realism",
                        "colors": cols,
                        "semantic_colors": cols,
                        "visual_details": [raw_content[:150].strip()],
                        "possible_camera_characteristics": ["Standard photographic perspective"],
                        "generation_prompt": raw_content[:300].strip()
                    }

                # Validate and auto-complete required fields
                self._validate_schema(parsed)
                # Record which model actually responded
                parsed["_resolved_model"] = m_name
                return parsed

            except ModelOutputValidationError as e:
                err_str = str(e)
                # Auth failures: short-circuit (won't be fixed by trying another model)
                if "Authentication failed" in err_str or "invalid API key" in err_str:
                    raise
                # All other structured errors: log and try next model
                print(f"[LlamaProvider] {err_str}")
                last_err = e
                continue
            except requests.exceptions.Timeout:
                err_msg = f"{m_name} request timed out after {t_out}s (NIM public queue congested)"
                print(f"[LlamaProvider] {err_msg}")
                last_err = ModelOutputValidationError(err_msg)
                continue
            except requests.exceptions.ConnectionError as e:
                err_msg = f"{m_name} connection error: {e}"
                print(f"[LlamaProvider] {err_msg}")
                last_err = ModelOutputValidationError(err_msg)
                continue
            except Exception as e:
                err_msg = f"{m_name} unexpected error: {type(e).__name__}: {e}"
                print(f"[LlamaProvider] {err_msg}")
                last_err = ModelOutputValidationError(err_msg)
                continue

        raise ModelOutputValidationError(f"All Llama vision models failed. Last error: {str(last_err)}")





    def _validate_schema(self, data: Dict[str, Any]):
        """Ensures all essential fields exist; validates required fields and completes minor sub-fields."""
        if not isinstance(data, dict) or not data:
            raise ModelOutputValidationError("Invalid Llama output: response must be a non-empty dictionary.")
        if not any(k in data and data[k] for k in ("scene", "summary", "detailed_description", "generation_prompt")):
            raise ModelOutputValidationError("Invalid Llama output: missing essential descriptive fields (scene/summary).")

        if not data.get("scene"):
            data["scene"] = data.get("summary") or "Visual scene"
        if not data.get("lighting"):
            data["lighting"] = "Natural lighting"
        if not data.get("composition"):
            data["composition"] = "Balanced framing"
        if not data.get("generation_prompt"):
            data["generation_prompt"] = f"Photorealistic depiction of {data.get('scene', 'scene')}, {data.get('lighting', 'natural lighting')}, {data.get('composition', 'balanced framing')}"

llama_vision_provider = LlamaVisionProvider()
