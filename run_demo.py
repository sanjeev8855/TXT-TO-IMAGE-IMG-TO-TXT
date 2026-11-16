import io
import json
import requests
from PIL import Image, ImageDraw

BASE_URL = "http://127.0.0.1:8000"

def run_demonstration():
    print("=" * 70)
    print("MULTIMODAL AI & EXPLAINABLE FORENSICS: LIVE RESEARCH PIPELINE DEMO")
    print("=" * 70)

    # Health Check
    health = requests.get(f"{BASE_URL}/health").json()
    print(f"\n[Status Check] Server Health: {health['status']}")
    print(f"  VLM Model: {health['vlm_model']}")
    print(f"  Forensic Model: {health['forensic_model']}")

    # 1. CREATE & UPLOAD INPUT IMAGE
    print("\n[STAGE 1: IMAGE INPUT & PREPROCESSING]")
    # Synthesize an input scene: coastal landscape with sunset
    w, h = 400, 300
    img = Image.new("RGB", (w, h), color=(240, 160, 90))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, int(h * 0.6), w, h], fill=(30, 80, 140)) # Ocean
    draw.ellipse([int(w * 0.4), int(h * 0.3), int(w * 0.6), int(h * 0.55)], fill=(255, 230, 160)) # Sun
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("demo_sunset.png", buf.getvalue(), "image/png")}
    upload_res = requests.post(f"{BASE_URL}/api/upload", files=files).json()
    filename = upload_res["filename"]
    print(f"  Uploaded File: {filename}")
    print(f"  Resolution: {upload_res['processed_dimensions'][0]}x{upload_res['processed_dimensions'][1]} px")
    print(f"  Mean Intensity: {upload_res['mean_intensity']:.3f}")
    print(f"  EXIF Stripped: {upload_res['exif_stripped']}")

    # 2. VLM IMAGE -> TEXT
    print("\n[STAGE 2: VLM IMAGE -> TEXT (VISUAL UNDERSTANDING)]")
    vlm_res = requests.post(f"{BASE_URL}/api/image-to-text", json={"filename": filename}).json()
    print(f"  Model Provider: {vlm_res['model_provider']}")
    print(f"  Scene: {vlm_res['scene']}")
    print(f"  Lighting: {vlm_res['lighting']}")
    print(f"  Composition: {vlm_res['composition']}")
    print(f"  Colors: {', '.join(vlm_res['colors'])}")
    print("  Direct Observations vs. Inferred Assumptions:")
    print(f"    - Observations: {vlm_res['observations_vs_assumptions'].get('direct_observations')}")
    print(f"    - Inferences: {vlm_res['observations_vs_assumptions'].get('contextual_assumptions')}")

    # 3. RAG RETRIEVAL
    print("\n[STAGE 3: RAG RETRIEVAL (DOMAIN KNOWLEDGE)]")
    query = f"{vlm_res['scene']} {vlm_res['lighting']} {vlm_res['composition']}"
    rag_res = requests.post(f"{BASE_URL}/api/rag-retrieve", json={"query": query, "top_k": 3}).json()
    print(f"  Query: '{query}'")
    for doc in rag_res["retrieved_documents"]:
        print(f"  -> [{doc['category'].upper()}] {doc['title']} (Cosine: {doc['similarity_score']}): {doc['description']}")

    # 4. LLM PROMPT BUILDER & BOUNDARY LAYER
    print("\n[STAGE 4: PROMPT CONSTRUCTION & BOUNDARY VALIDATION LAYER]")
    prompt_res = requests.post(f"{BASE_URL}/api/prompt-build", json={
        "vlm_output": vlm_res,
        "retrieved_documents": rag_res["retrieved_documents"],
        "user_instructions": "cinematic photorealism, high aesthetic quality"
    }).json()
    print(f"  Prompt Validated & Safe: {prompt_res['is_valid']}")
    print(f"  Sanitization Applied: {prompt_res['sanitization_applied']}")
    print("  Structured 10-Part Breakdown:")
    for k, v in prompt_res["structured_sections"].items():
        if v:
            print(f"    - {k.upper()}: {v}")
    print(f"\n  FINAL SANITIZED PROMPT:\n  \"{prompt_res['sanitized_prompt']}\"")

    # 5. TEXT -> IMAGE GENERATION
    print("\n[STAGE 5: TEXT-TO-IMAGE LATENT DIFFUSION]")
    gen_res = requests.post(f"{BASE_URL}/api/text-to-image", json={
        "prompt": prompt_res["sanitized_prompt"],
        "seed": 42,
        "steps": 30
    }).json()
    print(f"  Synthesized Image: {gen_res['image_filename']}")
    print(f"  Generator Engine: {gen_res['provider']}")
    print("  Process Execution Timeline:")
    for step in gen_res["process_timeline"]:
        print(f"    [OK] {step['name']} ({step['detail']}) - {step['duration_ms']}ms")

    # 6. FORENSIC AI DETECTOR
    print("\n[STAGE 6: INDEPENDENT FORENSIC DETECTION & EXPLAINABILITY (XAI)]")
    forensic_res = requests.post(f"{BASE_URL}/api/forensics/analyze", json={
        "filename": gen_res["image_filename"],
        "alpha": 0.45
    }).json()

    rep = forensic_res["report"]
    vis = forensic_res["visualizations"]
    print(f"  Report ID: {rep['report_id']}")
    print(f"  Evaluated Forensic Model: {rep['forensic_model']}")
    print(f"  AI-Generated Likelihood: {rep['ai_generated_likelihood']}% (Model Assessment)")
    print(f"  Real-Image Likelihood: {rep['real_image_likelihood']}%")
    print(f"  Forensic Anomaly Level: {rep['anomaly_level']} (Top-15% Mean: {rep['global_anomaly_score']})")
    print(f"  Patches Evaluated: {rep['patch_statistics']['total_patches_evaluated']} (64x64 px, stride 32)")
    print("\n  Forensic Visualizations Generated:")
    print(f"    - Standalone Heatmap: {vis['heatmap_url']}")
    print(f"    - Heatmap Overlay: {vis['overlay_url']}")
    if "fft_url" in vis:
        print(f"    - 2D Fourier Power Spectrum: {vis['fft_url']}")
    if "residual_url" in vis:
        print(f"    - Spatial Noise Residual: {vis['residual_url']}")

    print("\n  Forensic Explanation:")
    print(f"    \"{rep['forensic_explanation']}\"")
    print("\n  Academic Notice:")
    print(f"    \"{rep['academic_disclaimer']}\"")

    print("\n" + "=" * 70)
    print("DEMONSTRATION RUN COMPLETED SUCCESSFULLY")
    print(f"Visit the Web Dashboard at: {BASE_URL}")
    print("=" * 70)

if __name__ == "__main__":
    run_demonstration()
