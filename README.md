# Multimodal Image-to-Text, RAG-Enhanced Text-to-Image Generation and Explainable AI Image Forensics

**An Academic Research Prototype & Experimental Laboratory**

---

## 1. System Overview

This software prototype provides a rigorous, explainable, and reproducible research workbench demonstrating two bidirectional multimodal capabilities coupled with an independent computer-vision forensic analysis module:

1. **IMAGE → TEXT (Visual Understanding & Model Routing):** 
   - **Primary Vision:** Google Gemini Vision extracts rich visual context (scene, lighting, subjects, foreground/background, atmosphere).
   - **Primary Image → Text:** Llama 3.2 90B Vision Instruct receives the original image, Gemini vision context, and computer-vision pixel color extraction, returning an enriched, structured 10+ attribute JSON representation.
   - **Fallback Model:** Kimi K3 takes over automatically if Llama 3.2 90B fails (network error, timeout, rate limit, authentication error, JSON parse failure, or schema validation failure).
2. **RAG RETRIEVAL & PROMPT REASONING:** 
   - Vector retrieval over domain-specific visual and photography knowledge bases (optics, lighting, composition, camera systems, styles, quality descriptors).
   - Prompt synthesis and refinement powered by Kimi K3 / Gemini LLM reasoning through a strict Prompt Boundary & Validation Layer defending against prompt injection.
3. **TEXT → IMAGE (Iterative Latent Synthesis):** Denoising latent diffusion with step-by-step telemetry (encoding, latent initialization, reverse diffusion schedule, and spatial VAE decoding) with optional Google Imagen 3 support.
4. **INDEPENDENT FORENSIC DETECTION (XAI):** A completely decoupled multi-stream forensic analyzer evaluating overlapping 64×64 spatial patches using discrete spatial noise residuals (Laplacian high-pass filter), 2D Fourier (FFT) power spectra, and texture variance. Converts the 2D anomaly matrix into an infrared pseudo-thermal heatmap and alpha-blended overlay for human-in-the-loop explainability.

> **Academic Notice:** This software is an academic research prototype. Forensic anomaly scores quantify statistical artifact discrepancies and model confidence; they do **not** claim mathematical certainty or absolute legal proof of synthetic or authentic origin.

---

## 2. High-Level Architectural Workflow & Model Routing

```
                          IMAGE INPUT
                               │
                               ▼
                    IMAGE PREPROCESSING (RGB, EXIF Sanitized)
                               │
                               ▼
                    PRIMARY VISION: GEMINI VISION
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │  PRIMARY IMAGE → TEXT: LLAMA 3.2 90B VISION  │
        └──────────────────────┬───────────────────────┘
                               │
                  Success? ────┴──── Error / Schema Failure?
                   │                                │
                   ▼                                ▼
        [STRUCTURED DESCRIPTION]         FALLBACK: KIMI K3
                   │                                │
                   └───────────────┬────────────────┘
                                   │
                                   ▼
                         RAG RETRIEVAL & REASONING (Kimi K3)
                                   │
                                   ▼
                        PROMPT BOUNDARY SANITIZATION
                                   │
                                   ▼
                      TEXT-TO-IMAGE GENERATION (Diffusion / Imagen)
                                   │
                                   ▼
                               IMAGE OUTPUT
                                   │
                                   ▼
                  INDEPENDENT FORENSIC DETECTOR (Decoupled)
                 (Laplacian Noise + 2D FFT + Patch Variance)
                                   │
                                   ▼
                 PATCH ANOMALY MAP & PSEUDO-THERMAL HEATMAP
                                   │
                                   ▼
                             FORENSIC REPORT
```

---

## 3. Modular System Architecture

| Module | Subsystem | File Location | Responsibility |
|---|---|---|---|
| **Module 1** | Image Input & Preprocessing | `backend/image_to_text/preprocessing.py` | Validates file format, size (max 20MB), dimensions (64px–4096px), auto-orients via EXIF, converts to normalized RGB. |
| **Module 2** | Color & Pixel Analysis | `backend/image_to_text/preprocessing.py` | OpenCV + Scikit-Learn $K$-Means clustering; extracts exact HEX, RGB, percentages, and warm/cool classifications directly from pixels. |
| **Module 3** | Vision & Model Routing | `backend/image_to_text/vlm.py` | Orchestrates Gemini Vision → Llama 3.2 90B Vision → Kimi K3 fallback; outputs comprehensive structured JSON. |
| **Module 4** | Model Providers | `backend/providers/` | Abstract provider layer (`GeminiProvider`, `LlamaProvider`, `KimiProvider`, `Factory`). |
| **Module 5** | RAG Retrieval System | `backend/rag/` | 256-D dense vector embedding & cosine store over 33 photography & lighting domain documents. |
| **Module 6** | Prompt Builder & Reasoning | `backend/rag/prompt_builder.py` | Kimi K3 / Gemini LLM reasoning synthesizes VLM data + RAG context + user instructions into a 10-part schema. |
| **Module 7** | Prompt Boundary Layer | `backend/rag/prompt_builder.py` | Sanitizes prompt injections, shell commands, script tags, and length boundaries. |
| **Module 8** | Text-to-Image Generator | `backend/text_to_image/generator.py` | Latent diffusion denoising with 5-stage telemetry and research metadata (or Google Imagen 3). |
| **Module 9** | Image Storage & Logger | `backend/utils/research_logger.py` | SQLite & JSON experiment logger tracking audit trails, parameters, and scores. |
| **Module 10** | Forensic AI Detector | `backend/forensic/detector.py` | Decoupled multi-stream forensic analyzer (Noise residual, 2D FFT, local variance). |
| **Module 11** | Patch-Level Analysis | `backend/forensic/patch_extractor.py` | Extracts 64×64 overlapping spatial patches (stride 32). |
| **Module 12** | Anomaly Map Generator | `backend/forensic/anomaly_map.py` | Bicubic continuous 2D spatial interpolation matching image dimensions. |
| **Module 13** | Pseudo-Thermal Heatmap | `backend/forensic/heatmap.py` | Infrared colormap (Blue → Green → Yellow → Red) and alpha blending. |
| **Module 14** | Forensic Report Generator | `backend/forensic/report.py` | Robust top-15% percentile aggregation, likelihood calibration, and disclaimers. |
| **Module 15** | Frontend Dashboard | `frontend/index.html`, `frontend/app.js` | Interactive academic laboratory UI with real-time model routing audit, fallback simulator, and live telemetry. |

---

## 4. Environment Configuration (`.env`)

Copy `.env.example` to `.env`:
```powershell
copy .env.example .env
```

Configure your API keys as desired:
```env
# Google Gemini Vision & LLM
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_VISION_MODEL=gemini-2.5-flash
GEMINI_LLM_MODEL=gemini-2.5-flash

# Primary Image-to-Text: Llama 3.2 90B Vision Instruct
# Compatible with Groq (gsk_...), Together AI, OpenRouter (sk-or-...), or local vLLM/Ollama
LLAMA_API_KEY=your_llama_or_groq_api_key_here
LLAMA_MODEL=llama-3.2-90b-vision-instruct
# LLAMA_API_BASE=https://api.groq.com/openai/v1

# Fallback & Prompt Reasoning: Kimi K3 (Moonshot AI)
KIMI_API_KEY=your_kimi_api_key_here
KIMI_MODEL=kimi-k3
# KIMI_API_BASE=https://api.moonshot.cn/v1
```

*(If API keys are not provided, the system gracefully falls back to local computer-vision extraction and local latent synthesis without throwing unhandled exceptions or presenting fabricated mock data).*

---

## 5. Running the Application

### Start the Server
```powershell
python backend/main.py
```
Or directly with Uvicorn:
```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### Access Dashboard & Documentation
- **Web Application Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive OpenAPI Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **System Status Endpoint:** [http://127.0.0.1:8000/api/system-status](http://127.0.0.1:8000/api/system-status)
- **Health Check Endpoint:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 6. Model Routing & Fallback Verification

### In the Web UI:
Under **Stage 1: Image Preprocessing & Input**, check the **"Simulate Llama failure (Test Kimi K3 fallback)"** checkbox, then click **Execute Multimodal Pipeline**.
The **Model Pipeline Status** card in Stage 2 will display:
- **Vision:** `✓ Gemini Vision (gemini-2.5-flash)`
- **Primary:** `✗ Llama 3.2 90B Vision`
- **Fallback:** `✓ Kimi K3 (Fallback)`
- **Actual Model Used:** `Kimi K3` (or local fallback if offline)
- **Latency & Audit Reason:** Transparently displayed in the UI.

### Automated Unit & Integration Tests:
Run the complete test suite:
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Run the specific 6 model routing verification tests:
```powershell
python -m unittest tests/test_model_routing.py
```
The test suite validates:
1. Normal pipeline execution.
2. Forced Llama failure → Kimi K3 fallback activated.
3. Invalid Llama response schema → schema validation failure → Kimi K3 fallback.
4. Kimi failure → controlled error handling without crash.
5. Independent Forensic Analysis operation.
6. Forensic PDF/text report generation.

---

## 7. Research Methodology & Forensic Signals

### Multi-Stream Forensic Signals
1. **Spatial Noise Residual Stream (High-Pass Laplacian Filter):**  
   Natural camera sensors feature Poisson-Gaussian shot noise distributions governed by physical photon arrival statistics and Bayer filter sensor PRNU. In contrast, generative diffusion decoders exhibit spatially non-uniform residual variances and characteristic smoothing.
2. **Frequency Domain 2D Fourier (FFT) Power Spectrum:**  
   Latent diffusion models upsample 4-channel latents using convolutional transpose or nearest-neighbor blocks that introduce periodic high-frequency grid harmonics (checkerboard artifacts) visible as spectral spikes in the 2D power spectrum.
3. **Patch-Level Aggregation:**  
   Overlapping 64×64 patches with stride 32 ensure full boundary coverage. Image-level likelihood is calculated via the mean of the top-15% percentile anomaly scores, preventing isolated high-contrast natural edges from distorting the overall assessment.

### Pseudo-Thermal Heatmap Colormap
- **Blue (0.0):** Baseline / low detected anomaly.
- **Green / Yellow (0.5):** Moderate artifact presence.
- **Red / White (1.0):** Strong detected generative model artifact evidence.
