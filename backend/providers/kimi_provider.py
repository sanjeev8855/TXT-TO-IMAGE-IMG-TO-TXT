"""
Kimi K3 Fallback Multimodal & Reasoning Provider (Section 5 & 7)
Provides automatic fallback for Image-to-Text when Llama 3.2 90B fails,
and acts as secondary reasoning provider for RAG prompt enhancement.
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

class KimiProviderError(Exception):
    """Raised when Kimi K3 fails or returns an unparseable response."""
    pass

def _format_list(val: Any) -> str:
    if isinstance(val, list):
        items = [str(x).strip() for x in val if x is not None and str(x).strip()]
        return ", ".join(items) if items else "N/A"
    if isinstance(val, str) and val.strip():
        return val.strip()
    return "N/A"

class KimiProvider:
    """
    Kimi K3 provider supporting Moonshot / Kimi OpenAI-compatible REST API.
    Used for Image -> Text fallback and RAG prompt enhancement reasoning.
    """
    def __init__(self):
        self.api_key = config.kimi_api_key
        self.model = config.kimi_model or "kimi-k3"
        self.api_base = (config.kimi_api_base or "https://api.moonshot.cn/v1").rstrip("/")

    @property
    def provider_name(self) -> str:
        return "Kimi K3 (Fallback Multimodal Provider)"

    def generate_structured_description(
        self,
        pil_image: Image.Image,
        gemini_context: Optional[Dict[str, Any]] = None,
        cv_color_data: Optional[Dict[str, Any]] = None,
        fallback_reason: str = "Primary model failure",
        force_fail: bool = False
    ) -> Dict[str, Any]:
        """
        Executes Kimi K3 fallback inference when Llama 3.2 90B fails.
        """
        if force_fail or config.force_kimi_failure:
            raise KimiProviderError("Simulated Kimi K3 fallback failure for controlled error evaluation.")

        if not self.api_key:
            raise KimiProviderError("KIMI_API_KEY is not configured on the server.")

        # Encode image to optimized lightweight base64 buffer for rapid NIM transport
        thumb = pil_image.copy()
        thumb.thumbnail((512, 512))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=82)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        gemini_summary = ""
        if gemini_context:
            gemini_summary = f"""
VISUAL CONTEXT FROM GEMINI STAGE:
- Scene: {gemini_context.get('scene', 'N/A')}
- Primary Subjects: {_format_list(gemini_context.get('subjects'))}
- Lighting: {gemini_context.get('lighting', 'N/A')}
- Composition: {gemini_context.get('composition', 'N/A')}
- Background: {gemini_context.get('environment', 'N/A')}
"""

        cv_summary = ""
        if cv_color_data:
            cols = [f"{c['name']} ({c['hex']}, {c['percentage']}%)" for c in cv_color_data.get("dominant_colors", [])[:5]]
            cv_summary = f"\nVERIFIED COMPUTER-VISION PIXEL COLORS:\n{', '.join(cols)}\n"

        prompt_instruction = f"""You are Kimi K3, a senior multimodal vision-language research assistant taking over in fallback mode.
Primary reason for fallback activation: {fallback_reason}.
Analyze this image and synthesize a validated structured image description.
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
Return raw valid JSON without markdown wrapping.
"""

        endpoint = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Attempt 1: Vision mode (image + text) if model supports it
        # Attempt 2: Text-only mode using Gemini context (graceful degradation)
        for attempt in ("vision", "text_only"):
            if attempt == "vision":
                content_parts = [
                    {"type": "text", "text": prompt_instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            else:
                # Text-only: inject image description from Gemini visual context
                # Kimi K3 on NIM may not support vision or may be congested
                vision_desc = ""
                if gemini_context:
                    vision_desc = (
                        f"\n\n[IMAGE CONTENT — derived from Gemini Vision analysis since direct image unavailable]\n"
                        f"Scene: {gemini_context.get('scene', 'N/A')}\n"
                        f"Subjects: {_format_list(gemini_context.get('subjects'))}\n"
                        f"Lighting: {gemini_context.get('lighting', 'N/A')}\n"
                        f"Composition: {gemini_context.get('composition', 'N/A')}\n"
                        f"Background: {gemini_context.get('background', gemini_context.get('environment', 'N/A'))}\n"
                        f"Atmosphere: {gemini_context.get('atmosphere', 'N/A')}\n"
                    )
                content_parts = [{"type": "text", "text": prompt_instruction + vision_desc}]
                print(f"[KimiProvider] Switching to text-only mode (vision attempt failed or timed out)")

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": content_parts
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 1000
            }

            try:
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=10)
                if resp.status_code == 401:
                    raise KimiProviderError("Kimi K3 authentication failed: invalid API key")
                if resp.status_code == 400:
                    if attempt == "vision":
                        print(f"[KimiProvider] Vision mode rejected (400), retrying text-only")
                        continue
                    raise KimiProviderError(f"Kimi K3 bad request (400): {resp.text[:200]}")
                if resp.status_code == 422:
                    if attempt == "vision":
                        print(f"[KimiProvider] Vision mode rejected (422), retrying text-only")
                        continue
                    raise KimiProviderError(f"Kimi K3 model rejected request (422)")
                if resp.status_code == 429:
                    raise KimiProviderError("Kimi K3 rate limit exceeded (429)")
                if resp.status_code != 200:
                    raise KimiProviderError(f"HTTP {resp.status_code} from Kimi K3: {resp.text[:200]}")

                res_json = resp.json()
                choices = res_json.get("choices", [])
                if not choices:
                    raise KimiProviderError("Empty choices list returned from Kimi K3")

                raw_content = choices[0]["message"]["content"].strip()
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

                parsed = json.loads(clean_json_str)
                parsed["_kimi_mode"] = attempt
                return parsed

            except KimiProviderError:
                raise
            except requests.exceptions.Timeout:
                err = f"Kimi K3 timed out in {attempt} mode (35s)"
                print(f"[KimiProvider] {err}")
                if attempt == "vision":
                    continue  # try text_only
                raise KimiProviderError(err)
            except requests.exceptions.ConnectionError as e:
                raise KimiProviderError(f"Kimi K3 connection error: {e}")
            except json.JSONDecodeError as e:
                err = f"Kimi K3 returned invalid JSON in {attempt} mode: {e}"
                print(f"[KimiProvider] {err}")
                if attempt == "vision":
                    continue
                raise KimiProviderError(err)
            except Exception as e:
                err = f"Kimi K3 unexpected error in {attempt} mode: {type(e).__name__}: {e}"
                print(f"[KimiProvider] {err}")
                if attempt == "vision":
                    continue
                raise KimiProviderError(err)

        raise KimiProviderError("Kimi K3 failed in both vision and text-only modes")



    def enhance_prompt_reasoning(
        self,
        rag_context_docs: List[Dict[str, Any]],
        structured_desc: Dict[str, Any],
        user_instructions: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Uses Kimi K3 for the prompt-enhancement reasoning stage (Section 7):
        RAG context + structured image description -> Kimi K3 -> enhanced generation prompt.
        """
        if not self.api_key or config.force_kimi_failure:
            raise KimiProviderError("KIMI_API_KEY is not configured or disabled.")

        rag_text = "\n".join([f"- {d['category'].upper()}: {d['title']} - {d['description']}" for d in rag_context_docs])
        prompt_input = f"""You are Kimi K3, a reasoning prompt-engineering specialist for high-fidelity latent diffusion.
Synthesize the structured image breakdown, retrieved visual RAG knowledge, and user directives into a cohesive 10-part generation prompt.

Structured Visual Analysis:
- Scene: {structured_desc.get('scene', '')}
- Subjects: {_format_list(structured_desc.get('subjects'))}
- Lighting: {structured_desc.get('lighting', '')}
- Composition: {structured_desc.get('composition', '')}
- Environment: {structured_desc.get('environment', '')}
- Colors: {_format_list(structured_desc.get('colors'))}

Retrieved Domain Knowledge (RAG):
{rag_text}

User Directives:
{user_instructions or 'Preserve natural realism and high optical fidelity.'}

Return JSON with exactly these 10 keys:
"subject", "environment", "composition", "lighting", "camera_view", "style", "color", "material_texture", "detail", "quality".
Each value must be an expressive visual phrase. Output strictly raw JSON.
"""

        endpoint = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt_input}],
            "temperature": 0.35,
            "max_tokens": 800
        }

        resp = requests.post(endpoint, headers=headers, json=payload, timeout=8)
        if resp.status_code != 200:
            raise KimiProviderError(f"HTTP {resp.status_code} from Kimi reasoning: {resp.text[:300]}")

        res_json = resp.json()
        raw = res_json["choices"][0]["message"]["content"].strip()
        clean_json_str = raw
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if fence_match:
            clean_json_str = fence_match.group(1).strip()
        else:
            first_b = raw.find("{")
            last_b = raw.rfind("}")
            if first_b != -1 and last_b != -1 and last_b > first_b:
                clean_json_str = raw[first_b : last_b + 1].strip()

        return json.loads(clean_json_str)

kimi_provider = KimiProvider()
