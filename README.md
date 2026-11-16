# Multimodal AI Visual Studio: Image-to-Text, RAG-Enhanced Text-to-Image & Explainable AI Forensics

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![NVIDIA GenAI](https://img.shields.io/badge/NVIDIA%20NIM-FLUX.1--dev-76B900.svg?logo=nvidia&logoColor=white)](https://build.nvidia.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![QA Test Suite](https://img.shields.io/badge/Tests-14%2F14%20PASS-brightgreen.svg)](scripts/test_full_api.py)

An end-to-end multimodal research studio and explainable AI workbench. Features bidirectional vision understanding (Image → Text via Vision-Language Models), domain-specialized Retrieval-Augmented Generation (RAG) with optical conditioning, high-fidelity neural latent diffusion (Text → Image via NVIDIA FLUX.1-dev and SDXL), and a completely decoupled spatial-frequency patch-level forensic detector with infrared thermal heatmaps.

---

## Table of Contents

- [1. System Architecture](#1-system-architecture)
- [2. Core Pillars & Key Capabilities](#2-core-pillars--key-capabilities)
  - [Pillar 1: Multi-Model Vision & VLM Cascading (Image → Text)](#pillar-1-multi-model-vision--vlm-cascading-image--text)
  - [Pillar 2: Optical Knowledge RAG & Prompt Boundary Layer](#pillar-2-optical-knowledge-rag--prompt-boundary-layer)
  - [Pillar 3: Neural Latent Diffusion & NVIDIA GenAI (Text → Image)](#pillar-3-neural-latent-diffusion--nvidia-genai-text--image)
  - [Pillar 4: Decoupled Explainable AI (XAI) Forensics](#pillar-4-decoupled-explainable-ai-xai-forensics)
  - [Pillar 5: Bi-directional Evaluation & Parity Metrics](#pillar-5-bi-directional-evaluation--parity-metrics)
  - [Pillar 6: Interactive 3D Spatial UI & Glassmorphism](#pillar-6-interactive-3d-spatial-ui--glassmorphism)
- [3. Project Roadmap](#3-project-roadmap)
- [4. Project Structure](#4-project-structure)
- [5. API Endpoints Reference](#5-api-endpoints-reference)
- [6. Getting Started & Installation](#6-getting-started--installation)
- [7. Configuration (.env)](#7-configuration-env)
- [8. Automated Testing & Verification](#8-automated-testing--verification)
- [9. Academic & Ethical Disclaimer](#9-academic--ethical-disclaimer)

---

## 1. System Architecture

```
                                  USER IMAGE INPUT
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │  Stage 1: Preprocessing & Security    │
                     │  - EXIF Data Sanitization             │
                     │  - Pixel K-Means Color Distribution   │
                     │  - Aspect Ratio & Dimension Rescaling │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │  Stage 2: Vision-Language Routing     │
                     │  - Primary: Gemini Vision / Llama 90B │
                     │  - Fallback: Kimi K3 / CV Heuristics  │
                     │  - Output: 10+ Facet Structured JSON  │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │  Stage 3: RAG Retrieval & Reasoning   │
                     │  - 33 Optical & Lighting Knowledge DB │
                     │  - Dense Cosine Vector Similarity     │
                     │  - Prompt Boundary Sanitization Layer │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │  Stage 4: Latent Diffusion Synthesis  │
                     │  - Primary: NVIDIA NIM (FLUX.1-dev)   │
                     │  - Fallbacks: Pollinations / Imagen 3 │
                     │  - 5-Step Progress Telemetry          │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │  Stage 5: Bi-directional Parity & XAI │
                     │  - Color & Semantic Parity Comparison │
                     │  - Decoupled Patch Forensic Detector  │
                     │  - High-Pass Laplacian Noise Residual │
                     │  - 2D Fourier (FFT) Power Harmonics   │
                     │  - Pseudo-Thermal Heatmap & Overlay   │
                     └───────────────────────────────────────┘
```

---

## 2. Core Pillars & Key Capabilities

### Pillar 1: Multi-Model Vision & VLM Cascading (Image → Text)
- **Pixel-Level Color Analysis:** Direct pixel-space color quantization using OpenCV and Scikit-Learn $K$-Means clustering. Extracts dominant colors, percentage distributions, HEX codes, and warm/cool color balance directly from image pixels.
- **Cascading Model Routing:**
  - **Primary:** Google Gemini Vision (`gemini-2.5-flash` / `gemini-3.5-flash`) extracts rich visual semantics.
  - **Structured Reasoning:** Meta Llama 3.2 90B Vision Instruct generates a calibrated 10+ attribute JSON description (scene composition, lighting quality, focal subject, optical characteristics, atmosphere, etc.).
  - **Automatic Fallback:** Gracefully fails over to Kimi K3 or an offline computer-vision heuristic pipeline if rate limits, timeouts, or network errors occur.

### Pillar 2: Optical Knowledge RAG & Prompt Boundary Layer
- **33 Curated Optical Documents:** Indexed domain knowledge spanning camera sensors, aperture depth of field, 35mm/wide-angle focal lengths, golden hour volumetric lighting, and photographic composition.
- **Dense Cosine Vector Store:** High-speed vectorized cosine similarity matching that retrieves the most relevant physical principles for the detected scene.
- **Prompt Boundary Defense:** Strict adversarial filtering that eliminates prompt injection attempts, system instruction overrides, markdown/script injections, and length boundaries.

### Pillar 3: Neural Latent Diffusion & NVIDIA GenAI (Text → Image)
- **NVIDIA GenAI NIM (`black-forest-labs/flux.1-dev`):** High-resolution 1024×1024 photorealistic synthesis using state-of-the-art flow matching latent diffusion.
- **Multi-Tier Redundancy:** Automatic multi-tier failover:
  1. NVIDIA NIM FLUX.1-dev
  2. Pollinations SDXL Realism Engine
  3. Google Imagen 3 Cloud API
  4. AI Horde Distributed GPU Cluster
  5. Deterministic Procedural Academic Latent Synthesizer
- **5-Stage Diffusion Telemetry:** Tracks execution time and progress across Prompt Tokenization, Latent Sampling, Reverse Denoising, Spatial VAE Decoding, and Dynamic Range Calibration.

### Pillar 4: Decoupled Explainable AI (XAI) Forensics
- **Strict Architecture Decoupling:** The forensic analysis engine operates independently from generative synthesis models to prevent evaluation bias.
- **Multi-Stream Spatial & Frequency Patch Analysis:**
  - **Laplacian High-Pass Filter:** Detects non-uniform pixel variance and unnatural smoothing in high-frequency noise residuals.
  - **2D Fast Fourier Transform (FFT):** Identifies periodic grid artifacts and checkerboard harmonics introduced during convolutional upsampling.
  - **Local Patch Variance:** Evaluates overlapping 64×64 spatial patches across the entire image with stride 32.
- **Top-15% Robust Percentile Metric:** Computes the image-level anomaly score from the 85th percentile (top 15% highest patch scores), preventing isolated high-contrast natural edges from triggering false positives.
- **Pseudo-Thermal Heatmaps:** Renders continuous bicubic-interpolated anomaly heatmaps (Blue $\to$ Cyan $\to$ Yellow $\to$ Red) with alpha-blended overlays.

### Pillar 5: Bi-directional Evaluation & Parity Metrics
- **Color Parity:** Quantitative comparison of color histograms between original input and generated output.
- **Semantic Consistency:** Cross-modal semantic similarity verifying that key visual entities, lighting mood, and spatial layouts are preserved.
- **Audit Logging:** Every experiment and comparison is stored in a structured SQLite database (`research_experiments.db`) with full JSON telemetry.

### Pillar 6: Interactive 3D Spatial UI & Glassmorphism
- **60 FPS Canvas 2D 3D Perspective Projection:** 75 interconnected neural nodes and floating light particles rendered in 3D perspective space (`#bg-3d-canvas`).
- **Mouse Parallax Physics:** Smoothly tilts and rotates in real time with cursor movement via damped lerp physics.
- **Luminous Aurora Mesh:** Multi-spectral keyframe-animated glow orbs shifting behind panels.
- **Frosted Glass Cards:** Glassmorphic `.res-card` containers with micro-depth elevation (`backdrop-filter: blur(14px)`).
- **Unrestricted Scrolling:** Canvas set to `pointer-events: none` to guarantee 100% free vertical page scrolling.

---

## 3. Project Roadmap

```
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                               PROJECT ROADMAP                               │
  └─────────────────────────────────────────────────────────────────────────────┘

  [✓] Phase 1: Core Multimodal Foundation
      ├── VLM pipeline (Gemini Vision + Llama 3.2 90B)
      ├── OpenCV K-Means pixel color distribution extraction
      ├── SQLite experiment logging & audit database
      └── Baseline spatial forensic patch detector

  [✓] Phase 2: Domain Knowledge RAG & Security
      ├── 33 Curated optical, lighting, and camera documents
      ├── Dense vector cosine similarity retrieval
      └── Strict Prompt Boundary Layer preventing prompt injections

  [✓] Phase 3: High-Fidelity Diffusion & Multi-Provider Architecture
      ├── NVIDIA GenAI NIM FLUX.1-dev (1024×1024 photorealistic synthesis)
      ├── Pollinations SDXL fallback with clean inpainting & de-watermarking
      ├── Google Imagen 3 cloud API integration
      └── Distributed AI Horde GPU cluster connector

  [✓] Phase 4: Explainable Forensics & Bi-directional Parity
      ├── High-pass Laplacian noise residual stream
      ├── 2D FFT spectral harmonic checkerboard detection
      ├── Bicubic pseudo-thermal infrared anomaly heatmap & overlay
      └── Bi-directional semantic & color parity scoring engine

  [✓] Phase 5: Modern Visual UI & Interactive 3D Canvas
      ├── 3D spatial particle constellation with real-time mouse parallax
      ├── Floating multi-spectral aurora glow mesh
      ├── Glassmorphic card elevation and radiant button shimmers
      └── Dark / light theme persistence with zero-scroll interference

  [ ] Phase 6: Future Milestones (Upcoming)
      ├── Video-to-Text & temporal frame-to-frame forensic consistency
      ├── Multi-agent autonomous verification pipeline
      ├── Docker & Kubernetes deployment configurations
      └── WebAssembly (Wasm) client-side forensic patch inspection
```

---

## 4. Project Structure

```
TXT-TO-IMAGE-IMG-TO-TXT/
├── .env.example                # Sample environment configuration template
├── .gitignore                  # Git exclusions (.env, runtime images, pycache)
├── config.yaml                 # Core system configuration and model defaults
├── package.json                # Project metadata
├── README.md                   # Comprehensive documentation & architecture guide
├── requirements.txt            # Python dependencies
├── research_experiments.db     # SQLite database storing experiment logs
├── run_demo.py                 # Standalone CLI demo of the pipeline
├── run_server.bat              # Windows batch file to launch the server
│
├── backend/                    # Core Python Backend (FastAPI)
│   ├── main.py                 # FastAPI application entrypoint & static mounting
│   ├── config.py               # Dynamic settings, directory creation & env loader
│   ├── api/
│   │   ├── routes.py           # REST API endpoints (upload, VLM, RAG, diffusion, forensics)
│   │   └── schemas.py          # Pydantic validation schemas
│   ├── image_to_text/
│   │   ├── preprocessing.py    # Image validation, EXIF sanitization & resizing
│   │   ├── color_extractor.py  # OpenCV K-Means dominant color extraction
│   │   ├── object_detector.py  # Spatial entity & bounding box extraction
│   │   ├── scene_analyzer.py   # High-level scene classification
│   │   └── vlm.py              # Multi-model VLM orchestrator & fallback router
│   ├── providers/
│   │   ├── base.py             # Abstract base provider interface
│   │   ├── factory.py          # Provider factory and model registry
│   │   ├── gemini_provider.py  # Google Gemini Vision & LLM integration
│   │   ├── llama_provider.py   # Meta Llama 3.2 90B Vision Instruct integration
│   │   └── kimi_provider.py    # Moonshot Kimi K3 provider
│   ├── rag/
│   │   ├── embeddings.py       # Dense text embedding generator
│   │   ├── vector_store.py     # In-memory cosine vector store
│   │   ├── retriever.py        # Top-K semantic knowledge retriever
│   │   ├── knowledge_builder.py# Document loader and indexer
│   │   └── prompt_builder.py   # Optical prompt reasoning & boundary sanitizer
│   ├── text_to_image/
│   │   └── generator.py        # Latent diffusion generator (NVIDIA FLUX, SDXL, Imagen)
│   ├── forensic/
│   │   ├── detector.py         # Multi-stream forensic patch detector
│   │   ├── patch_extractor.py  # 64x64 overlapping spatial patch generator
│   │   ├── anomaly_map.py      # 2D bicubic spatial anomaly interpolation
│   │   ├── heatmap.py          # Pseudo-thermal infrared colormap & overlay
│   │   └── report.py           # Formal forensic evaluation report generator
│   └── utils/
│       └── research_logger.py  # Structured experiment & audit logging
│
├── data/
│   └── rag_documents/          # 33 Optical & Photographic Domain Knowledge Docs
│       ├── art_styles.json
│       ├── camera_optics.json
│       ├── composition.json
│       ├── forensic_context.json
│       ├── lighting.json
│       ├── photography.json
│       └── quality_descriptors.json
│
├── frontend/                   # Frontend Web Application
│   ├── index.html              # Single-page laboratory dashboard with 3D canvas
│   └── app.js                  # Frontend state, API connector & 3D parallax engine
│
├── outputs/                    # Runtime outputs (.gitignore managed)
│   ├── generated/              # Synthesized diffusion images (.gitkeep)
│   ├── uploads/                # Sanitized user uploads (.gitkeep)
│   ├── heatmaps/               # Forensic thermal heatmaps & overlays (.gitkeep)
│   └── reports/                # JSON forensic evaluation reports (.gitkeep)
│
├── scripts/
│   └── test_full_api.py        # Comprehensive 14-endpoint automated test suite
│
└── tests/                      # Unit & Integration Tests
    ├── test_end_to_end.py      # Complete pipeline test
    ├── test_forensic_detector.py# Forensic patch detector tests
    ├── test_model_routing.py   # VLM fallback & failure simulation tests
    ├── test_preprocessing.py   # Image EXIF & color extraction tests
    ├── test_prompt_builder.py  # Prompt injection & boundary defense tests
    ├── test_rag.py             # Knowledge retrieval tests
    └── test_text_to_image.py   # Latent diffusion generator tests
```

---

## 5. API Endpoints Reference

| Method | Endpoint | Description | Key Parameters |
|---|---|---|---|
| `GET` | `/health` | System health check | Returns status, uptime, and active provider info |
| `GET` | `/api/system-status` | Comprehensive status of all models & RAG docs | Returns RAG document count and provider status |
| `POST` | `/api/upload` | Upload & sanitize input image | `file` (Multipart image file, max 20MB) |
| `POST` | `/api/image-to-text` | Run VLM extraction & color analysis | `image_path`, `simulate_llama_failure` |
| `POST` | `/api/rag-retrieve` | Retrieve optical knowledge from vector store | `query`, `top_k` (default 5) |
| `POST` | `/api/prompt-build` | Construct sanitized diffusion prompt | `vlm_data`, `rag_context`, `user_directive` |
| `POST` | `/api/text-to-image` | Synthesize image via latent diffusion / FLUX | `prompt`, `seed`, `steps`, `provider` |
| `POST` | `/api/forensics/analyze` | Run decoupled patch-level forensic analysis | `image_path`, `patch_size` (64), `stride` (32) |
| `POST` | `/api/bidirectional/compare` | Evaluate color & semantic parity | `original_image_path`, `generated_image_path` |
| `GET` | `/api/experiments` | List experiment audit history from database | `limit` (default 25) |
| `GET` | `/api/reports` | List generated forensic evaluation reports | None |
| `GET` | `/api/knowledge-base` | Retrieve indexed optical knowledge corpus | None |

---

## 6. Getting Started & Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git installed on your system
- An NVIDIA NIM, Google Gemini, or compatible API key (optional for cloud generation; local fallbacks operate offline)

### Step 1: Clone the Repository
```bash
git clone https://github.com/sanjeev8855/TXT-TO-IMAGE-IMG-TO-TXT.git
cd TXT-TO-IMAGE-IMG-TO-TXT
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
```bash
# Copy sample template
copy .env.example .env     # Windows
cp .env.example .env       # Linux / macOS
```
Edit `.env` with your preferred API keys (see [Configuration](#7-configuration-env)).

### Step 5: Start the Server
```bash
python backend/main.py
```
Or on Windows, simply double-click **`run_server.bat`**.

Open your browser at **`http://127.0.0.1:8000`** to access the visual studio.

---

## 7. Configuration (.env)

```env
# Google Gemini Vision & LLM
GEMINI_API_KEY=your_gemini_api_key_here
VISION_MODEL=gemini-2.5-flash

# Meta Llama 3.2 90B Vision Instruct
LLAMA_API_KEY=your_llama_or_groq_api_key_here
LLAMA_MODEL=llama-3.2-90b-vision-instruct
LLAMA_API_BASE=https://api.groq.com/openai/v1

# Moonshot Kimi K3 (Fallback & Reasoning)
KIMI_API_KEY=your_kimi_api_key_here
KIMI_MODEL=kimi-k3
KIMI_API_BASE=https://api.moonshot.cn/v1

# NVIDIA GenAI / FLUX.1-dev Latent Diffusion
IMAGE_GENERATION_PROVIDER=nvidia
IMAGE_GENERATION_API_KEY=your_nvapi_key_here
IMAGE_GENERATION_MODEL=black-forest-labs/flux.1-dev

# Testing & Simulation Controls
FORCE_LLAMA_FAILURE=false
FORCE_KIMI_FAILURE=false

# Server Settings
HOST=127.0.0.1
PORT=8000
DEBUG=True
```

---

## 8. Automated Testing & Verification

The repository includes a comprehensive 14-endpoint end-to-end test suite verifying every component from health checks to forensic heatmap generation:

```bash
# Run the complete test suite
python scripts/test_full_api.py
```

### Expected Output:
```
  FINAL QA REPORT
  ------------------------------------------------------------
  [PASS] health_endpoint
  [PASS] system_status
  [PASS] image_upload
  [PASS] image_to_text
  [PASS] rag_retrieval
  [PASS] prompt_builder
  [PASS] text_to_image
  [PASS] generated_image_accessible
  [PASS] forensic_analysis
  [PASS] heatmap_generated
  [PASS] bidirectional_compare
  [PASS] experiments_log
  [PASS] knowledge_base
  [PASS] reports_listing

  Results: 14/14 PASS, 0 WARN, 0 FAIL
```

Run standard unit tests:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 9. Academic & Ethical Disclaimer

> **Academic Research Prototype:** This software is designed for academic, educational, and research evaluation. The forensic anomaly detector measures statistical discrepancies in spatial frequency and noise distributions; it provides probabilistic indicator confidence and does **not** constitute mathematical certainty or definitive legal evidence of image provenance.
