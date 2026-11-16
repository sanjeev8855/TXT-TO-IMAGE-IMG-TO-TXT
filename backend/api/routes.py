import os
import shutil
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from PIL import Image

from backend.config import config
from backend.image_to_text.preprocessing import preprocess_image, ImagePreprocessingError
from backend.image_to_text.vlm import vlm_service, VLMStructuredOutput
from backend.rag.retriever import rag_retriever
from backend.rag.vector_store import vector_store
from backend.rag.prompt_builder import prompt_builder
from backend.text_to_image.generator import image_generator
from backend.forensic.detector import forensic_detector
from backend.forensic.anomaly_map import anomaly_map_generator
from backend.forensic.heatmap import heatmap_generator
from backend.forensic.report import forensic_report_generator
from backend.utils.research_logger import research_logger
from backend.api.schemas import (
    SystemStatusResponse,
    ImageUploadResponse,
    VLMAnalysisRequest,
    RAGRetrieveRequest,
    RAGRetrieveResponse,
    PromptBuildRequest,
    TextToImageRequest,
    ForensicAnalysisRequest,
    FullPipelineRequest,
    BidirectionalCompareRequest,
)

router = APIRouter(prefix="/api", tags=["Research Prototype API"])

def _find_image_path(filename: str) -> Path:
    """Finds image in uploads, generated, or mirrored directories safely."""
    import shutil
    clean_name = Path(filename).name
    # 1. Primary directories
    for p in [config.uploads_dir / clean_name, config.generated_dir / clean_name, config.output_dir / clean_name]:
        if p.exists():
            return p

    # 2. Desktop & OneDrive mirror directories
    home = Path(os.path.expanduser("~"))
    search_dirs = [
        home / "Desktop" / "IMAGE TO TEXT CONVTER" / "outputs" / "uploads",
        home / "Desktop" / "IMAGE TO TEXT CONVTER" / "outputs" / "generated",
        home / "OneDrive" / "Desktop" / "IMAGE TO TEXT CONVTER" / "outputs" / "uploads",
        home / "OneDrive" / "Desktop" / "IMAGE TO TEXT CONVTER" / "outputs" / "generated",
        home / "Desktop" / "IMAGE TO TEXT CONVTER",
        home / "OneDrive" / "Desktop" / "IMAGE TO TEXT CONVTER",
    ]
    for d in search_dirs:
        candidate = d / clean_name
        if candidate.exists() and candidate.is_file():
            target = config.uploads_dir / clean_name
            try:
                shutil.copy2(candidate, target)
                return target
            except Exception:
                return candidate

    raise HTTPException(status_code=404, detail=f"Image file '{clean_name}' not found on server.")

@router.get("/system-status", response_model=SystemStatusResponse)
async def get_system_status():
    from backend.providers.factory import get_all_provider_statuses
    return SystemStatusResponse(
        models=config.models,
        gemini_api_configured=bool(config.gemini_api_key),
        reproducibility=config.reproducibility,
        rag_document_count=len(vector_store.documents),
        forensic_config=config.forensics,
        model_providers=get_all_provider_statuses()
    )

@router.post("/upload", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """Module 1 & 2: Handles image upload, security verification, and EXIF-stripped preprocessing."""
    try:
        contents = await file.read()
        pil_img, np_arr, meta = preprocess_image(contents)

        # Save to uploads dir
        safe_name = f"upload_{uuid.uuid4().hex[:10]}_{Path(file.filename or 'img.png').name}"
        save_path = config.uploads_dir / safe_name
        pil_img.save(save_path, format="PNG")

        return ImageUploadResponse(
            filename=safe_name,
            file_url=f"/outputs/uploads/{safe_name}",
            original_dimensions=[meta["original_width"], meta["original_height"]],
            processed_dimensions=[meta["processed_width"], meta["processed_height"]],
            format=meta["format"],
            mean_intensity=meta["mean_intensity"],
            std_intensity=meta["std_intensity"],
            exif_stripped=meta["exif_stripped"]
        )
    except ImagePreprocessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@router.post("/image-to-text", response_model=VLMStructuredOutput)
def run_image_to_text(req: VLMAnalysisRequest):
    """Module 3 & 4: Vision-Language Model visual understanding -> structured representation."""
    img_path = _find_image_path(req.filename)
    try:
        with Image.open(img_path) as raw_img:
            img = raw_img.convert("RGB")
            structured_desc = vlm_service.analyze_image(
                pil_image=img,
                force_llama_fail=getattr(req, "force_llama_failure", False),
                force_kimi_fail=getattr(req, "force_kimi_failure", False)
            )
            return structured_desc
    except Exception as e:
        print(f"[API Image-to-Text Error] Processing '{req.filename}': {e}")
        try:
            with Image.open(img_path) as raw_img:
                img = raw_img.convert("RGB")
                return vlm_service._fallback_structured_output(
                    pil_image=img,
                    np_image=None,
                    summary=f"Visual features and color distribution extracted from {Path(req.filename).name}",
                    gen_prompt=f"Photorealistic visual scene with natural lighting, 8k resolution"
                )
        except Exception as inner_e:
            raise HTTPException(status_code=500, detail=f"Visual analysis failed: {str(e)}")

@router.post("/rag-retrieve", response_model=RAGRetrieveResponse)
def run_rag_retrieve(req: RAGRetrieveRequest):
    """Module 4: Vector similarity search against domain visual knowledge."""
    try:
        docs = rag_retriever.retrieve_context(
            query_text=req.query,
            top_k=req.top_k,
            filter_category=req.category_filter
        )
        summary = rag_retriever.build_context_summary(docs)
        return RAGRetrieveResponse(
            query=req.query,
            top_k=req.top_k or config.rag.get("top_k", 5),
            retrieved_documents=docs,
            context_summary=summary
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG retrieval error: {str(e)}")

@router.post("/prompt-build")
def run_prompt_build(req: PromptBuildRequest):
    """Module 5 & 6: LLM Prompt Builder and Prompt Boundary Validation Layer."""
    try:
        res = prompt_builder.build_prompt(
            vlm_desc=req.vlm_output,
            retrieved_docs=req.retrieved_documents,
            user_instructions=req.user_instructions
        )
        return res.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prompt builder error: {str(e)}")

@router.post("/text-to-image")
def run_text_to_image(req: TextToImageRequest):
    """Module 7: Latent Denoising and Image Synthesis with progress timeline."""
    try:
        result = image_generator.generate_image(
            prompt=req.prompt,
            user_seed=req.seed,
            custom_steps=req.steps
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@router.post("/forensics/analyze")
def run_forensic_analysis(req: ForensicAnalysisRequest):
    """
    Modules 9, 10, 11, 12, 13:
    Independent Forensic AI Detector -> Patch scores -> Anomaly map -> Pseudo-thermal Heatmap -> Report.
    """
    img_path = _find_image_path(req.filename)
    try:
        with Image.open(img_path) as raw_img:
            rgb_img = raw_img.convert("RGB")
            import numpy as np
            np_arr = np.array(rgb_img, dtype=np.float32) / 255.0

            # 1. Patch extraction & classifier
            eval_res = forensic_detector.evaluate_image(np_arr)
            score_mat = eval_res["score_matrix"]

            # 2. Continuous Anomaly Map with multi-scale micro-residual fusion
            cont_map = anomaly_map_generator.generate_continuous_map(
                score_mat,
                (rgb_img.height, rgb_img.width),
                high_res_residual=eval_res.get("noise_residual_array")
            )

            # 3. Pseudo-thermal Heatmap & Overlay
            vis_paths = heatmap_generator.save_forensic_visualizations(
                orig_img=rgb_img,
                anomaly_map=cont_map,
                fft_array=eval_res.get("fft_spectrum_array"),
                residual_array=eval_res.get("noise_residual_array"),
                alpha=req.alpha
            )

            # Determine public URL for the original analyzed image
            clean_name = Path(req.filename).name
            if (config.uploads_dir / clean_name).exists():
                orig_url = f"/outputs/uploads/{clean_name}"
            elif (config.generated_dir / clean_name).exists():
                orig_url = f"/outputs/generated/{clean_name}"
            else:
                orig_url = f"/outputs/{clean_name}"
            vis_paths["orig_url"] = orig_url

            # 4. Forensic Report & Academic Disclaimers
            image_meta = {
                "filename": req.filename,
                "width": rgb_img.width,
                "height": rgb_img.height,
                "aspect_ratio": round(rgb_img.width / float(rgb_img.height), 2),
            }
            report = forensic_report_generator.generate_report(
                score_matrix=score_mat,
                image_meta=image_meta,
                vis_paths=vis_paths
            )

            # 5. Log experiment to SQLite metadata store
            research_logger.log_experiment(
                experiment_type="forensic_analysis",
                input_filename=req.filename,
                image_dims=(rgb_img.width, rgb_img.height),
                selected_models=config.models,
                rag_top_k=config.rag.get("top_k", 5),
                retrieved_doc_ids=[],
                final_prompt="",
                generation_params={"alpha": req.alpha},
                forensic_model=report["forensic_model"],
                global_anomaly_score=report["global_anomaly_score"],
                ai_generated_likelihood=report["ai_generated_likelihood"],
                anomaly_level=report["anomaly_level"],
                heatmap_path=vis_paths["heatmap_path"],
                overlay_path=vis_paths["overlay_path"],
                report_id=report["report_id"],
                extra_meta={"patch_count": eval_res["total_patches"]}
            )

            return {
                "report": report,
                "patch_scores": eval_res["patch_records"][:20],  # Sample for UI inspectability
                "patch_grid": {"rows": eval_res["n_rows"], "cols": eval_res["n_cols"]},
                "score_matrix_sample": score_mat.round(3).tolist(),
                "visualizations": vis_paths
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forensic analysis failed: {str(e)}")

@router.post("/pipeline/full-run")
async def run_full_pipeline(req: FullPipelineRequest):
    """
    End-to-end execution of the complete high-level workflow:
    IMAGE INPUT -> VLM -> RAG RETRIEVAL -> LLM PROMPT -> FINAL PROMPT ->
    DIFFUSION -> GENERATED IMAGE -> FORENSIC DETECTOR -> HEATMAP -> REPORT
    """
    # 1. Locate uploaded image
    img_path = _find_image_path(req.filename)

    with Image.open(img_path) as raw_img:
        # 2. VLM with Model Routing (Gemini Vision -> Llama 3.2 90B -> Kimi K3 Fallback)
        vlm_output = vlm_service.analyze_image(
            raw_img,
            force_llama_fail=getattr(req, "force_llama_failure", False),
            force_kimi_fail=getattr(req, "force_kimi_failure", False)
        )

        # 3. RAG Retrieval
        query = f"{vlm_output.scene} {vlm_output.lighting} {vlm_output.composition} {', '.join(vlm_output.subjects)}"
        rag_docs = rag_retriever.retrieve_context(query, top_k=req.rag_top_k or 5)

        # 4. LLM Prompt Construction & Boundary Validation
        prompt_res = prompt_builder.build_prompt(vlm_output, rag_docs, req.user_instructions)

        # 5. Text-to-Image Generation
        gen_res = image_generator.generate_image(
            prompt=prompt_res.sanitized_prompt,
            user_seed=req.generation_seed,
            custom_steps=req.generation_steps
        )

        # 6. Forensic Detection on GENERATED image
        gen_filename = gen_res["image_filename"]
        gen_img_path = Path(gen_res["image_path"])

        with Image.open(gen_img_path) as gen_img:
            import numpy as np
            np_arr = np.array(gen_img.convert("RGB"), dtype=np.float32) / 255.0
            eval_res = forensic_detector.evaluate_image(np_arr)
            score_mat = eval_res["score_matrix"]
            cont_map = anomaly_map_generator.generate_continuous_map(
                score_mat,
                (gen_img.height, gen_img.width),
                high_res_residual=eval_res.get("noise_residual_array")
            )

            vis_paths = heatmap_generator.save_forensic_visualizations(
                orig_img=gen_img,
                anomaly_map=cont_map,
                fft_array=eval_res.get("fft_spectrum_array"),
                residual_array=eval_res.get("noise_residual_array")
            )

            report = forensic_report_generator.generate_report(
                score_matrix=score_mat,
                image_meta={"filename": gen_filename, "width": gen_img.width, "height": gen_img.height},
                vis_paths=vis_paths
            )

            # Log complete pipeline experiment
            doc_ids = [d["id"] for d in rag_docs]
            research_logger.log_experiment(
                experiment_type="full_pipeline_run",
                input_filename=req.filename,
                image_dims=(gen_img.width, gen_img.height),
                selected_models=config.models,
                rag_top_k=len(rag_docs),
                retrieved_doc_ids=doc_ids,
                final_prompt=prompt_res.sanitized_prompt,
                generation_params={"seed": gen_res["seed"], "steps": gen_res["steps"]},
                forensic_model=report["forensic_model"],
                global_anomaly_score=report["global_anomaly_score"],
                ai_generated_likelihood=report["ai_generated_likelihood"],
                anomaly_level=report["anomaly_level"],
                heatmap_path=vis_paths["heatmap_path"],
                overlay_path=vis_paths["overlay_path"],
                report_id=report["report_id"],
                extra_meta={
                    "source_upload": req.filename,
                    "model_routing": {
                        "vision_model": vlm_output.vision_model,
                        "primary_image_to_text_model": vlm_output.primary_image_to_text_model,
                        "actual_image_to_text_model": vlm_output.actual_image_to_text_model,
                        "fallback_used": vlm_output.fallback_used,
                        "fallback_reason": vlm_output.fallback_reason,
                        "latency_ms": vlm_output.latency_ms
                    }
                }
            )

            return {
                "stage1_uploaded_image": f"/outputs/uploads/{Path(req.filename).name}",
                "stage2_vlm_analysis": vlm_output.model_dump(),
                "stage3_rag_retrieval": {
                    "query": query,
                    "documents": rag_docs
                },
                "stage4_prompt_builder": prompt_res.model_dump(),
                "stage5_generated_image": gen_res,
                "stage6_forensic_analysis": {
                    "report": report,
                    "visualizations": vis_paths,
                    "patch_sample": eval_res["patch_records"][:16]
                }
            }

@router.get("/experiments")
async def get_experiments(limit: int = Query(default=30, ge=1, le=100)):
    """Module 8: Returns history of all logged research experiments."""
    return research_logger.get_recent_experiments(limit=limit)

@router.get("/knowledge-base")
async def get_knowledge_base():
    """Browse indexed RAG knowledge documents."""
    return [d.to_dict() for d in vector_store.documents]

@router.get("/reports")
async def get_reports():
    """Returns list of all generated forensic reports with summary statistics."""
    reports = []
    if config.reports_dir.exists():
        for p in sorted(config.reports_dir.glob("REP_*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    reports.append({
                        "report_id": data.get("report_id", p.stem),
                        "filename": data.get("image_metadata", {}).get("filename", "Unknown"),
                        "ai_generated_likelihood": data.get("ai_generated_likelihood", 0.0),
                        "real_image_likelihood": data.get("real_image_likelihood", 0.0),
                        "anomaly_level": data.get("anomaly_level", "UNKNOWN"),
                        "global_anomaly_score": data.get("global_anomaly_score", 0.0),
                        "forensic_model": data.get("forensic_model", "Unknown"),
                        "timestamp": p.stat().st_mtime,
                        "report_url": f"/outputs/reports/{p.name}"
                    })
            except Exception:
                continue
    return reports

@router.post("/bidirectional/compare")
async def compare_bidirectional(req: BidirectionalCompareRequest):
    """
    Computes genuine computer vision and NLP parity metrics:
    1. Color distribution correlation (OpenCV HSV histogram intersection / correlation).
    2. Semantic text embedding similarity (cosine distance of VLM description vs generation prompt).
    3. Resolution and aspect ratio parity.
    4. Combined multimodal consistency score.
    """
    import cv2
    import numpy as np
    from backend.rag.embeddings import embedding_engine

    orig_path = _find_image_path(req.original_filename)
    gen_path = _find_image_path(req.generated_filename)

    img1 = cv2.imread(str(orig_path))
    img2 = cv2.imread(str(gen_path))

    color_similarity = 0.5
    dim_match = False
    aspect_diff = 0.0

    if img1 is not None and img2 is not None:
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

        hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])

        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        raw_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        color_similarity = max(0.0, min(1.0, float((raw_corr + 1.0) / 2.0)))

        dim_match = (img1.shape[0] == img2.shape[0] and img1.shape[1] == img2.shape[1])
        ar1 = img1.shape[1] / max(1, img1.shape[0])
        ar2 = img2.shape[1] / max(1, img2.shape[0])
        aspect_diff = abs(ar1 - ar2)

    semantic_similarity = 0.5
    if req.original_description and req.generated_prompt:
        vec1 = embedding_engine.get_embedding(req.original_description)
        vec2 = embedding_engine.get_embedding(req.generated_prompt)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 > 1e-6 and norm2 > 1e-6:
            cos_sim = float(np.dot(vec1, vec2) / (norm1 * norm2))
            semantic_similarity = max(0.0, min(1.0, float((cos_sim + 1.0) / 2.0)))

    aspect_score = max(0.0, 1.0 - min(1.0, aspect_diff))
    overall_score = round((color_similarity * 0.4 + semantic_similarity * 0.4 + aspect_score * 0.2) * 100.0, 1)

    return {
        "color_distribution_correlation": round(color_similarity * 100.0, 1),
        "semantic_description_similarity": round(semantic_similarity * 100.0, 1),
        "aspect_ratio_parity": round(aspect_score * 100.0, 1),
        "overall_multimodal_consistency": overall_score,
        "original_dimensions": [int(img1.shape[1]), int(img1.shape[0])] if img1 is not None else None,
        "generated_dimensions": [int(img2.shape[1]), int(img2.shape[0])] if img2 is not None else None,
    }
