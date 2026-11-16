import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if available
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

CONFIG_PATH = BASE_DIR / "config.yaml"

DEFAULT_CONFIG = {
    "models": {
        "vlm_model": "gemini-3.5-flash",
        "vision_model": "gemini-3.5-flash",
        "primary_image_to_text_model": "llama-3.2-90b-vision-instruct",
        "fallback_image_to_text_model": "kimi-k3",
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "gemini-3.5-flash",
        "image_generation_model": "imagen-3.0-generate-002",
        "forensic_model": "spatial-frequency-patch-detector-v1",
    },
    "providers": {
        "vision": {"provider": "gemini", "model": "gemini-3.5-flash"},
        "image_to_text": {"provider": "llama", "model": "llama-3.2-90b-vision-instruct", "fallback": "kimi-k3"},
        "llm": {"provider": "gemini", "model": "gemini-3.5-flash"},
        "embedding": {"provider": "local", "model": "all-MiniLM-L6-v2"},
        "image_generation": {"provider": "imagen", "model": "imagen-3.0-generate-002"},
        "forensic": {"provider": "local", "model": "spatial-frequency-patch-detector-v1"},
    },
    "rag": {
        "top_k": 5,
        "similarity_threshold": 0.20,
        "knowledge_base_path": "data/rag_documents",
    },
    "prompt_boundary": {
        "max_length": 1000,
        "strict_validation": True,
        "forbidden_patterns": [
            r"(?i)(api[_-]?key|secret|password|bearer)",
            r"(?i)(exec\(|eval\(|system\(|os\.|\bsh\b|bash\b|cmd\.exe|powershell)",
            r"(?i)(<script|javascript:|SELECT .* FROM|INSERT INTO|DROP TABLE)",
            r"(?i)(ignore previous instructions|disregard all rules)",
        ],
    },
    "forensics": {
        "patch_size": 64,
        "stride": 32,
        "top_percentile": 0.15,
        "anomaly_threshold_low": 0.35,
        "anomaly_threshold_high": 0.65,
        "heatmap_colormap": "turbo",
        "heatmap_alpha": 0.45,
        "include_frequency_stream": True,
        "include_noise_residual_stream": True,
    },
    "reproducibility": {
        "random_seed": 42,
        "default_resolution": [512, 512],
        "generation_steps": 30,
        "guidance_scale": 7.5,
    },
}

class AppConfig:
    def __init__(self):
        self.reload()

    @staticmethod
    def _clean_token(val: str) -> str:
        if not val:
            return ""
        s = val.strip().strip('"').strip("'")
        if s.startswith("Bearer "):
            s = s[7:].strip().strip('"').strip("'")
        return s

    def reload(self):
        # Auto-discover .env from project root or Desktop folders
        env_candidates = [
            ENV_PATH,
            BASE_DIR / ".env.example",
            Path(os.path.expanduser("~")) / "Desktop" / "IMAGE TO TEXT CONVTER" / ".env",
            Path(os.path.expanduser("~")) / "Desktop" / "IMAGE TO TEXT CONVTER" / ".env.example",
            Path(os.path.expanduser("~")) / "OneDrive" / "Desktop" / "IMAGE TO TEXT CONVTER" / ".env",
            Path(os.path.expanduser("~")) / "OneDrive" / "Desktop" / "IMAGE TO TEXT CONVTER" / ".env.example",
        ]
        for p in env_candidates:
            if p.exists():
                load_dotenv(p, override=False)

        self.config_data = DEFAULT_CONFIG.copy()
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    user_yaml = yaml.safe_load(f)
                    if user_yaml and isinstance(user_yaml, dict):
                        self._deep_update(self.config_data, user_yaml)
            except Exception as e:
                print(f"[Config] Warning: Failed to parse config.yaml: {e}. Using defaults.")

        # Google Gemini API key (supports GEMINI_API_KEY or GOOGLE_API_KEY)
        self.gemini_api_key = self._clean_token(os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""))

        # Primary Vision
        self.vision_provider = os.getenv("VISION_PROVIDER", "gemini").strip()
        self.vision_api_key = self._clean_token(os.getenv("VISION_API_KEY", "")) or self.gemini_api_key
        self.vision_model = os.getenv("VISION_MODEL", self.config_data["models"].get("vision_model", "gemini-3.5-flash")).strip()
        if self.vision_model in ("gemini-2.5-flash", "gemini-2.0-flash"):
            self.vision_model = "gemini-3.5-flash"

        # Primary Image -> Text: Llama 3.2 90B Vision Instruct
        self.llama_api_key = self._clean_token(
            os.getenv("LLAMA_API_KEY", "") or
            os.getenv("GROQ_API_KEY", "") or
            os.getenv("TOGETHER_API_KEY", "") or
            os.getenv("OPENROUTER_API_KEY", "") or
            os.getenv("NVIDIA_API_KEY", "") or
            os.getenv("HF_TOKEN", "")
        )
        self.llama_model = os.getenv("LLAMA_MODEL", "llama-3.2-90b-vision-instruct").strip()
        self.llama_api_base = os.getenv("LLAMA_API_BASE", "").strip()

        # Auto-configure NVIDIA NIM if nvapi- key is detected
        if self.llama_api_key.startswith("nvapi-"):
            if not self.llama_api_base:
                self.llama_api_base = "https://integrate.api.nvidia.com/v1"
            if not self.llama_model.startswith("meta/"):
                self.llama_model = "meta/" + self.llama_model

        # Fallback Image -> Text & Prompt Reasoning: Kimi K3 (Moonshot / Kimi API / NVIDIA NIM)
        self.kimi_api_key = self._clean_token(os.getenv("KIMI_API_KEY", "") or os.getenv("MOONSHOT_API_KEY", ""))
        self.kimi_model = os.getenv("KIMI_MODEL", "kimi-k3").strip()
        self.kimi_api_base = os.getenv("KIMI_API_BASE", "").strip()

        if self.kimi_api_key.startswith("nvapi-"):
            if not self.kimi_api_base or "moonshot.cn" in self.kimi_api_base:
                self.kimi_api_base = "https://integrate.api.nvidia.com/v1"
            if self.kimi_model in ("kimi-k3", "kimi"):
                self.kimi_model = "moonshotai/kimi-k3"
        elif not self.kimi_api_base:
            self.kimi_api_base = "https://api.moonshot.cn/v1"

        # Simulation & Testing flags
        self.force_llama_failure = os.getenv("FORCE_LLAMA_FAILURE", "false").lower() in ("true", "1", "yes")
        self.force_kimi_failure = os.getenv("FORCE_KIMI_FAILURE", "false").lower() in ("true", "1", "yes")

        # LLM / Reasoning
        self.llm_provider = os.getenv("LLM_PROVIDER", "gemini").strip()
        self.llm_api_key = self._clean_token(os.getenv("LLM_API_KEY", "")) or self.gemini_api_key
        self.llm_model = os.getenv("LLM_MODEL", self.config_data["models"]["llm_model"]).strip()
        if self.llm_model in ("gemini-2.5-flash", "gemini-2.0-flash"):
            self.llm_model = "gemini-3.7-flash"

        # Embedding
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local").strip()
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY", "").strip() or self.gemini_api_key
        self.embedding_model = os.getenv("EMBEDDING_MODEL", self.config_data["models"]["embedding_model"]).strip()

        # Image Generation (Supports NVIDIA GenAI / FLUX.1-dev, Google Imagen 3, and AI Horde)
        self.image_generation_provider = os.getenv("IMAGE_GENERATION_PROVIDER", "nvidia").strip()
        self.image_generation_api_key = self._clean_token(
            os.getenv("IMAGE_GENERATION_API_KEY", "") or
            os.getenv("NETEON_API_KEY", "") or
            os.getenv("NVIDIA_IMAGE_API_KEY", "") or
            (self.llama_api_key if self.llama_api_key.startswith("nvapi-") else "") or
            self.gemini_api_key
        )
        self.image_generation_model = os.getenv("IMAGE_GENERATION_MODEL", "black-forest-labs/flux.1-dev").strip()
        self.neteon_api_key = self._clean_token(os.getenv("NETEON_API_KEY", "")) or self.image_generation_api_key
        self.nvidia_image_api_key = self._clean_token(os.getenv("NVIDIA_IMAGE_API_KEY", "")) or self.image_generation_api_key

        # Forensic Detector
        self.forensic_provider = os.getenv("FORENSIC_PROVIDER", "local").strip()
        self.forensic_api_key = os.getenv("FORENSIC_API_KEY", "").strip()
        self.forensic_model = os.getenv("FORENSIC_MODEL", self.config_data["models"]["forensic_model"]).strip()

        self.hf_token = os.getenv("HF_TOKEN", "").strip()
        self.host = os.getenv("HOST", "127.0.0.1")
        self.port = int(os.getenv("PORT", "8000"))
        self.debug = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

        # Directories
        self.base_dir = BASE_DIR
        self.data_dir = BASE_DIR / "data"
        self.rag_docs_dir = BASE_DIR / self.config_data["rag"]["knowledge_base_path"]
        self.output_dir = BASE_DIR / "outputs"
        self.uploads_dir = self.output_dir / "uploads"
        self.generated_dir = self.output_dir / "generated"
        self.heatmaps_dir = self.output_dir / "heatmaps"
        self.reports_dir = self.output_dir / "reports"

        for d in [self.uploads_dir, self.generated_dir, self.heatmaps_dir, self.reports_dir, self.rag_docs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _deep_update(self, base_dict, update_dict):
        for k, v in update_dict.items():
            if isinstance(v, dict) and k in base_dict and isinstance(base_dict[k], dict):
                self._deep_update(base_dict[k], v)
            else:
                base_dict[k] = v

    @property
    def models(self):
        return {
            "vlm_model": self.vision_model,
            "vision_model": self.vision_model,
            "primary_image_to_text_model": self.llama_model,
            "fallback_image_to_text_model": self.kimi_model,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "image_generation_model": self.image_generation_model,
            "forensic_model": self.forensic_model,
        }

    @property
    def providers(self):
        return {
            "vision": {"provider": "gemini", "configured": bool(self.vision_api_key), "model": self.vision_model},
            "primary_i2t": {"provider": "llama", "configured": bool(self.llama_api_key), "model": self.llama_model},
            "fallback_i2t": {"provider": "kimi", "configured": bool(self.kimi_api_key), "model": self.kimi_model},
            "llm": {"provider": self.llm_provider, "configured": bool(self.llm_api_key), "model": self.llm_model},
            "embedding": {"provider": self.embedding_provider, "configured": bool(self.embedding_api_key or self.embedding_provider == "local"), "model": self.embedding_model},
            "image_generation": {"provider": self.image_generation_provider, "configured": bool(self.image_generation_api_key or self.image_generation_provider == "local"), "model": self.image_generation_model},
            "forensic": {"provider": self.forensic_provider, "configured": bool(self.forensic_api_key or self.forensic_provider == "local"), "model": self.forensic_model},
        }

    @property
    def rag(self):
        return self.config_data.get("rag", {})

    @property
    def prompt_boundary(self):
        return self.config_data.get("prompt_boundary", {})

    @property
    def forensics(self):
        return self.config_data.get("forensics", {})

    @property
    def reproducibility(self):
        return self.config_data.get("reproducibility", {})

config = AppConfig()
