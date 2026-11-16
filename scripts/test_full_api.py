"""
Comprehensive end-to-end API test suite.
Tests every major endpoint on the running server.
Does NOT print API keys.
"""
import io
import time
import json
import base64
import requests
from pathlib import Path
from PIL import Image

BASE = "http://127.0.0.1:8000"
SEP = "-" * 60
results = {}


def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def PASS(label, detail=""):
    results[label] = "PASS"
    print(f"  [PASS] {label}: {detail}"[:120])


def FAIL(label, detail=""):
    results[label] = "FAIL"
    print(f"  [FAIL] {label}: {detail}"[:120])


def WARN(label, detail=""):
    results[label] = "WARN"
    print(f"  [WARN] {label}: {detail}"[:120])


# ── HEALTH / STATUS ──────────────────────────────────────────────────────────

section("TEST 1: Health endpoint")
try:
    r = requests.get(f"{BASE}/health", timeout=5)
    if r.status_code == 200 and r.json().get("status") == "healthy":
        PASS("health_endpoint", f"status=healthy, gemini={r.json().get('gemini_api_key_configured')}")
    else:
        FAIL("health_endpoint", f"HTTP {r.status_code}")
except Exception as e:
    FAIL("health_endpoint", str(e))

section("TEST 2: System status")
try:
    r = requests.get(f"{BASE}/api/system-status", timeout=5)
    if r.status_code == 200:
        d = r.json()
        providers = d.get("model_providers", {})
        rag_count = d.get("rag_document_count", 0)
        all_configured = all(p.get("configured") for p in providers.values())
        if rag_count > 0 and all_configured:
            PASS("system_status", f"rag_docs={rag_count}, all providers configured")
        else:
            WARN("system_status", f"rag_docs={rag_count}, providers={providers}")
    else:
        FAIL("system_status", f"HTTP {r.status_code}")
except Exception as e:
    FAIL("system_status", str(e))

# ── UPLOAD ───────────────────────────────────────────────────────────────────

section("TEST 3: Image upload")
uploaded_filename = None
try:
    # Create a simple test image
    img = Image.new("RGB", (256, 256), color=(80, 140, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    files = {"file": ("test_api.png", buf, "image/png")}
    r = requests.post(f"{BASE}/api/upload", files=files, timeout=30)
    if r.status_code == 200:
        d = r.json()
        uploaded_filename = d.get("filename")
        PASS("image_upload", f"filename={uploaded_filename}, dims={d.get('processed_dimensions')}")
    else:
        FAIL("image_upload", f"HTTP {r.status_code}: {r.text[:100]}")
except Exception as e:
    FAIL("image_upload", str(e))

# ── IMAGE TO TEXT ─────────────────────────────────────────────────────────────

section("TEST 4: Image-to-Text (VLM pipeline)")
vlm_output = None
if uploaded_filename:
    try:
        t0 = time.time()
        r = requests.post(f"{BASE}/api/image-to-text",
                          json={"filename": uploaded_filename},
                          timeout=120)
        elapsed = round(time.time() - t0, 2)
        if r.status_code == 200:
            vlm_output = r.json()
            model = vlm_output.get("model_provider", "?")
            actual = vlm_output.get("actual_image_to_text_model", "?")
            fallback = vlm_output.get("fallback_used", False)
            scene = vlm_output.get("scene", "?")[:60]
            gen_prompt = vlm_output.get("generation_prompt", "")[:80]
            PASS("image_to_text", f"model={model[:50]}, fallback={fallback}, {elapsed}s")
            print(f"    scene: {scene}")
            print(f"    gen_prompt: {gen_prompt}")
            print(f"    actual_model: {actual}")
        else:
            FAIL("image_to_text", f"HTTP {r.status_code}: {r.text[:150]}")
    except Exception as e:
        FAIL("image_to_text", str(e))
else:
    FAIL("image_to_text", "No uploaded file to analyze")

# ── RAG ───────────────────────────────────────────────────────────────────────

section("TEST 5: RAG retrieval")
rag_docs = []
try:
    query = vlm_output.get("scene", "ocean sunset") if vlm_output else "ocean sunset"
    t0 = time.time()
    r = requests.post(f"{BASE}/api/rag-retrieve",
                      json={"query": query, "top_k": 5},
                      timeout=10)
    elapsed = round(time.time() - t0, 2)
    if r.status_code == 200:
        d = r.json()
        rag_docs = d.get("retrieved_documents", [])
        PASS("rag_retrieval", f"{len(rag_docs)} docs, {elapsed}s")
        for doc in rag_docs[:2]:
            print(f"    [{doc.get('category')}] {doc.get('title','?')[:50]} score={doc.get('similarity_score',0):.3f}")
    else:
        FAIL("rag_retrieval", f"HTTP {r.status_code}: {r.text[:100]}")
except Exception as e:
    FAIL("rag_retrieval", str(e))

# ── PROMPT BUILD ──────────────────────────────────────────────────────────────

section("TEST 6: Prompt builder")
built_prompt = None
try:
    payload = {
        "vlm_output": vlm_output or {"scene": "test scene", "subjects": ["test"], "lighting": "natural"},
        "retrieved_documents": rag_docs,
        "user_instructions": "Preserve natural realism"
    }
    r = requests.post(f"{BASE}/api/prompt-build", json=payload, timeout=35)
    if r.status_code == 200:
        d = r.json()
        built_prompt = d.get("sanitized_prompt", "")
        PASS("prompt_builder", f"len={len(built_prompt)} chars, violations={d.get('violations_found', 0)}")
        print(f"    prompt[:100]: {built_prompt[:100]}")
    else:
        FAIL("prompt_builder", f"HTTP {r.status_code}: {r.text[:100]}")
except Exception as e:
    FAIL("prompt_builder", str(e))

# ── TEXT TO IMAGE ─────────────────────────────────────────────────────────────

section("TEST 7: Text-to-Image generation")
generated_filename = None
generated_url = None
try:
    use_prompt = built_prompt or "a photorealistic mountain landscape at golden hour"
    payload = {"prompt": use_prompt[:300], "seed": 42, "steps": 20}
    t0 = time.time()
    r = requests.post(f"{BASE}/api/text-to-image", json=payload, timeout=90)
    elapsed = round(time.time() - t0, 2)
    if r.status_code == 200:
        d = r.json()
        generated_filename = d.get("image_filename")
        generated_url = d.get("image_url")
        provider = d.get("provider", "?")
        resolution = d.get("resolution", [0, 0])
        PASS("text_to_image", f"provider={provider}, {resolution}, {elapsed}s")
        print(f"    filename: {generated_filename}")
        print(f"    url: {generated_url}")
        # Verify image is accessible
        if generated_url:
            img_r = requests.get(f"{BASE}{generated_url}", timeout=10)
            if img_r.status_code == 200 and len(img_r.content) > 1000:
                PASS("generated_image_accessible", f"{len(img_r.content)} bytes, Content-Type={img_r.headers.get('content-type')}")
            else:
                FAIL("generated_image_accessible", f"HTTP {img_r.status_code}, size={len(img_r.content)}")
    else:
        FAIL("text_to_image", f"HTTP {r.status_code}: {r.text[:200]}")
        FAIL("generated_image_accessible", "No image to check")
except Exception as e:
    FAIL("text_to_image", str(e))
    FAIL("generated_image_accessible", "Exception during generation")

# ── FORENSIC ANALYSIS ─────────────────────────────────────────────────────────

section("TEST 8: Forensic analysis")
forensic_file = generated_filename or uploaded_filename
if forensic_file:
    try:
        t0 = time.time()
        r = requests.post(f"{BASE}/api/forensics/analyze",
                          json={"filename": forensic_file, "alpha": 0.45},
                          timeout=30)
        elapsed = round(time.time() - t0, 2)
        if r.status_code == 200:
            d = r.json()
            report = d.get("report", {})
            score = report.get("global_anomaly_score", 0)
            level = report.get("anomaly_level", "?")
            likelihood = report.get("ai_generated_likelihood", 0)
            patches = report.get("total_patches_analyzed", 0)
            PASS("forensic_analysis", f"score={score:.3f}, level={level}, ai_likelihood={likelihood:.3f}, {elapsed}s")
            vis = d.get("visualizations", {})
            PASS("heatmap_generated", f"heatmap={vis.get('heatmap_path','?')[:50]}")
        else:
            FAIL("forensic_analysis", f"HTTP {r.status_code}: {r.text[:150]}")
            FAIL("heatmap_generated", "Forensics failed")
    except Exception as e:
        FAIL("forensic_analysis", str(e))
        FAIL("heatmap_generated", str(e))
else:
    FAIL("forensic_analysis", "No file to analyze")
    FAIL("heatmap_generated", "No file to analyze")

# ── BIDIRECTIONAL COMPARE ────────────────────────────────────────────────────

section("TEST 9: Bidirectional image comparison")
if uploaded_filename and generated_filename:
    try:
        payload = {
            "original_filename": uploaded_filename,
            "generated_filename": generated_filename,
            "original_description": vlm_output.get("detailed_description", "test image") if vlm_output else "test",
            "generated_prompt": built_prompt or "test prompt"
        }
        r = requests.post(f"{BASE}/api/bidirectional/compare", json=payload, timeout=15)
        if r.status_code == 200:
            d = r.json()
            overall = d.get("overall_multimodal_consistency", 0)
            color = d.get("color_distribution_correlation", 0)
            semantic = d.get("semantic_description_similarity", 0)
            PASS("bidirectional_compare", f"overall={overall}%, color={color}%, semantic={semantic}%")
        else:
            FAIL("bidirectional_compare", f"HTTP {r.status_code}: {r.text[:100]}")
    except Exception as e:
        FAIL("bidirectional_compare", str(e))
else:
    WARN("bidirectional_compare", "Need both uploaded and generated filenames")

# ── EXPERIMENTS LOG ───────────────────────────────────────────────────────────

section("TEST 10: Experiments log")
try:
    r = requests.get(f"{BASE}/api/experiments?limit=5", timeout=5)
    if r.status_code == 200:
        exps = r.json()
        PASS("experiments_log", f"{len(exps)} recent experiments in DB")
    else:
        FAIL("experiments_log", f"HTTP {r.status_code}")
except Exception as e:
    FAIL("experiments_log", str(e))

# ── KNOWLEDGE BASE ────────────────────────────────────────────────────────────

section("TEST 11: Knowledge base / RAG documents")
try:
    r = requests.get(f"{BASE}/api/knowledge-base", timeout=5)
    if r.status_code == 200:
        docs = r.json()
        PASS("knowledge_base", f"{len(docs)} RAG documents indexed")
    else:
        FAIL("knowledge_base", f"HTTP {r.status_code}")
except Exception as e:
    FAIL("knowledge_base", str(e))

# ── REPORTS ───────────────────────────────────────────────────────────────────

section("TEST 12: Forensic reports listing")
try:
    r = requests.get(f"{BASE}/api/reports", timeout=5)
    if r.status_code == 200:
        rpts = r.json()
        PASS("reports_listing", f"{len(rpts)} forensic reports")
    else:
        FAIL("reports_listing", f"HTTP {r.status_code}")
except Exception as e:
    FAIL("reports_listing", str(e))

# ── SUMMARY ───────────────────────────────────────────────────────────────────

section("FINAL QA REPORT")
passed = sum(1 for v in results.values() if v == "PASS")
warned = sum(1 for v in results.values() if v == "WARN")
failed = sum(1 for v in results.values() if v == "FAIL")
total = len(results)

for k, v in results.items():
    icon = "PASS" if v == "PASS" else ("WARN" if v == "WARN" else "FAIL")
    print(f"  [{icon}] {k}")

print(f"\n  Results: {passed}/{total} PASS, {warned} WARN, {failed} FAIL")
print()
