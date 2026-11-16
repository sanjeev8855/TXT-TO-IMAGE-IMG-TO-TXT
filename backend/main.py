import os
import sys
from pathlib import Path

# Ensure project root is in Python sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from backend.config import config
from backend.api.routes import router as api_router

app = FastAPI(
    title="Multimodal Image-to-Text, RAG-Enhanced Text-to-Image Generation and Explainable AI Image Forensics",
    description="An Academic Research Prototype demonstrating bidirectional multimodal synthesis and explainable patch-level AI forensics.",
    version="1.0.0"
)

# Enable CORS for local research dashboard and extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Mount outputs directory for generated assets, heatmaps, and uploads
outputs_dir = config.output_dir
outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

# Mount frontend static directory
frontend_dir = config.base_dir / "frontend"
frontend_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "status": "online",
        "system": "Multimodal Image-to-Text, RAG-Enhanced Text-to-Image and Forensic AI Prototype",
        "api_docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gemini_api_key_configured": bool(config.gemini_api_key),
        "vlm_model": config.models.get("vlm_model"),
        "forensic_model": config.models.get("forensic_model"),
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[Server Error] {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Research Server Error: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    print(f"Starting Research Server on http://{config.host}:{config.port}...")
    uvicorn.run("backend.main:app", host=config.host, port=config.port, reload=config.debug)
