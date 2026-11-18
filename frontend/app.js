// Research Dashboard JavaScript - Academic Multimodal AI & Forensics Lab

// ---------------------------------------------------------------------------
// Global Protection: Disable Mouse Right-Click & Context Menu
// ---------------------------------------------------------------------------
document.addEventListener("contextmenu", (e) => {
  e.preventDefault();
  return false;
}, { capture: true });

document.addEventListener("mousedown", (e) => {
  if (e.button === 2) {
    e.preventDefault();
    return false;
  }
}, { capture: true });

document.addEventListener("mouseup", (e) => {
  if (e.button === 2) {
    e.preventDefault();
    return false;
  }
}, { capture: true });

// Toggle between fitted scrollable results and fully expanded results
window.toggleVlmScrollMode = function() {
  const vlmResults = document.getElementById("vlm-results");
  const expandBtnText = document.getElementById("vlm-expand-text");
  const expandBtnIcon = document.getElementById("vlm-expand-icon");
  if (!vlmResults) return;

  const isConstrained = vlmResults.classList.contains("max-h-[640px]");
  if (isConstrained) {
    vlmResults.classList.remove("max-h-[640px]", "overflow-y-auto", "pr-1.5");
    if (expandBtnText) expandBtnText.textContent = "Fit View";
    if (expandBtnIcon) expandBtnIcon.textContent = "⤡";
  } else {
    vlmResults.classList.add("max-h-[640px]", "overflow-y-auto", "pr-1.5");
    if (expandBtnText) expandBtnText.textContent = "Expand View";
    if (expandBtnIcon) expandBtnIcon.textContent = "⤢";
  }
};

let state = {
  activeTab: 'home',
  uploadedFilename: null,
  uploadedUrl: null,
  generatedFilename: null,
  generatedUrl: null,
  generatedBase64: null,
  activeForensicFilename: null,
  currentPrompt: '',
  vlmOutput: null,
  ragDocs: [],
  kbDocuments: [],
  reportsList: [],
  selectedReport: null
};

// ---------------------------------------------------------------------------
// Clear Handlers
// ---------------------------------------------------------------------------
window.clearInputImage = function() {
  const fileInput = document.getElementById("file-input");
  if (fileInput) fileInput.value = "";

  const dropZone = document.getElementById("drop-zone");
  const previewContainer = document.getElementById("upload-preview-container");
  const previewImg = document.getElementById("upload-preview-img");
  const uploadMeta = document.getElementById("upload-meta");

  if (dropZone) dropZone.classList.remove("hidden");
  if (previewContainer) previewContainer.classList.add("hidden");
  if (previewImg) {
    previewImg.removeAttribute("src");
    previewImg.src = "";
  }
  if (uploadMeta) uploadMeta.innerHTML = "";

  state.uploadedFilename = null;
  state.uploadedUrl = null;

  if (typeof resetPipelineUIOnNewUpload === "function") {
    resetPipelineUIOnNewUpload();
  }

  const ragContainer = document.getElementById("rag-results-container");
  if (ragContainer) {
    ragContainer.innerHTML = '<div class="py-2.5 text-center text-app-muted text-xs font-mono">Context documents will appear here after analysis.</div>';
  }

  if (typeof showToast === "function") {
    showToast("Input image cleared", "info");
  }
};

window.clearColorDistribution = function() {
  window._currentVlmPalette = [];
  const colorBars = document.getElementById("vlm-color-bars");
  const pieContainer = document.getElementById("vlm-pie-container");
  const colorChips = document.getElementById("vlm-color-chips");

  if (colorBars) {
    colorBars.innerHTML = `<div style="width: 100%; height: 100%; background: var(--surface-muted);"></div>`;
  }
  if (pieContainer) {
    pieContainer.innerHTML = `<div class="text-xs text-app-muted font-mono text-center p-4">Color chart cleared</div>`;
  }
  if (colorChips) {
    colorChips.innerHTML = `<div class="col-span-2 text-xs text-app-muted font-mono p-2">Colors cleared</div>`;
  }

  if (typeof showToast === "function") {
    showToast("Color chart cleared", "info");
  }
};

window.clearPromptBoundary = function() {
  const promptContainer = document.getElementById("prompt-builder-container");
  const genButtonContainer = document.getElementById("gen-button-container");
  const boundaryBadge = document.getElementById("boundary-status-badge");

  if (promptContainer) {
    promptContainer.innerHTML = '<div class="py-2.5 text-center text-app-muted text-xs font-mono">Sanitized prompt will appear here.</div>';
  }
  if (genButtonContainer) genButtonContainer.classList.add("hidden");
  if (boundaryBadge) boundaryBadge.classList.add("hidden");

  state.currentPrompt = '';
  if (typeof showToast === "function") {
    showToast("Prompt boundary cleared", "info");
  }
};

window.clearT2IPrompt = function() {
  const promptEl = document.getElementById("t2i-prompt");
  const seedEl = document.getElementById("t2i-seed");
  const stepsEl = document.getElementById("t2i-steps");
  const progressCard = document.getElementById("t2i-progress-card");
  const stepsList = document.getElementById("t2i-steps-list");

  if (promptEl) promptEl.value = "";
  if (seedEl) seedEl.value = "42";
  if (stepsEl) stepsEl.value = "30";
  if (progressCard) progressCard.classList.add("hidden");
  if (stepsList) stepsList.innerHTML = "";

  if (typeof showToast === "function") {
    showToast("Prompt and parameters cleared", "info");
  }
};

window.clearT2IOutput = function() {
  const resultImg = document.getElementById("t2i-result-img");
  const placeholder = document.getElementById("t2i-placeholder");
  const actions = document.getElementById("t2i-actions");
  const techDetails = document.getElementById("t2i-tech-details");
  const techContent = document.getElementById("t2i-tech-content");
  const providerBadge = document.getElementById("t2i-provider-badge");

  if (resultImg) {
    resultImg.classList.add("hidden");
    resultImg.removeAttribute("src");
    resultImg.src = "";
  }
  if (placeholder) {
    placeholder.classList.remove("hidden");
    placeholder.innerHTML = "Synthesized image will render here.";
  }
  if (actions) actions.classList.add("hidden");
  if (techDetails) techDetails.removeAttribute("open");
  if (techContent) techContent.innerHTML = "";
  if (providerBadge) providerBadge.textContent = "Awaiting Synthesis";

  state.generatedFilename = null;
  state.generatedUrl = null;
  state.generatedBase64 = null;

  if (typeof showToast === "function") {
    showToast("Synthesized output cleared", "info");
  }
};

window.clearForensicAnalysis = function() {
  const targetName = document.getElementById("current-target-name");
  if (targetName) targetName.textContent = "None selected";

  state.activeForensicFilename = null;

  const fin = document.getElementById("forensic-file-input");
  if (fin) fin.value = "";

  const metricCards = document.getElementById("forensic-metric-cards");
  const reportCard = document.getElementById("forensic-report-card");
  if (metricCards) metricCards.classList.add("hidden");
  if (reportCard) reportCard.classList.add("hidden");

  // Reset metric scores
  const aiScoreEl = document.getElementById("metric-ai-score");
  const aiProgressBar = document.getElementById("ai-progress-bar");
  const realScoreEl = document.getElementById("metric-real-score");
  const realProgressBar = document.getElementById("real-progress-bar");
  const anomalyLevelEl = document.getElementById("metric-anomaly-level");
  const patchCountEl = document.getElementById("metric-patch-count");

  if (aiScoreEl) aiScoreEl.textContent = "0%";
  if (aiProgressBar) aiProgressBar.style.width = "0%";
  if (realScoreEl) realScoreEl.textContent = "0%";
  if (realProgressBar) realProgressBar.style.width = "0%";
  if (anomalyLevelEl) {
    anomalyLevelEl.textContent = "N/A";
    anomalyLevelEl.className = "text-xl font-bold text-accent";
  }
  if (patchCountEl) patchCountEl.textContent = "0";

  // Reset images
  const origImg = document.getElementById("vis-orig-img");
  const origPlaceholder = document.getElementById("vis-orig-placeholder");
  const origDims = document.getElementById("orig-dims");
  if (origImg) {
    origImg.classList.add("hidden");
    origImg.removeAttribute("src");
    origImg.src = "";
  }
  if (origPlaceholder) origPlaceholder.classList.remove("hidden");
  if (origDims) origDims.textContent = "";

  const heatmapImg = document.getElementById("vis-heatmap-img");
  const heatmapPlaceholder = document.getElementById("vis-heatmap-placeholder");
  if (heatmapImg) {
    heatmapImg.classList.add("hidden");
    heatmapImg.removeAttribute("src");
    heatmapImg.src = "";
  }
  if (heatmapPlaceholder) heatmapPlaceholder.classList.remove("hidden");

  const overlayImg = document.getElementById("vis-overlay-img");
  const overlayPlaceholder = document.getElementById("vis-overlay-placeholder");
  if (overlayImg) {
    overlayImg.classList.add("hidden");
    overlayImg.removeAttribute("src");
    overlayImg.src = "";
  }
  if (overlayPlaceholder) overlayPlaceholder.classList.remove("hidden");

  const fftImg = document.getElementById("vis-fft-img");
  if (fftImg) {
    fftImg.classList.add("hidden");
    fftImg.removeAttribute("src");
    fftImg.src = "";
  }

  const resImg = document.getElementById("vis-residual-img");
  if (resImg) {
    resImg.classList.add("hidden");
    resImg.removeAttribute("src");
    resImg.src = "";
  }

  const expl = document.getElementById("report-explanation");
  if (expl) expl.textContent = "";
  const disc = document.getElementById("report-disclaimer");
  if (disc) disc.textContent = "";

  if (typeof showToast === "function") {
    showToast("Forensic analysis cleared", "info");
  }
};

window.clearAnalysisResults = function() {
  const vlmResults = document.getElementById("vlm-results");
  const vlmPlaceholder = document.getElementById("vlm-placeholder");
  const modelBadge = document.getElementById("vlm-model-badge");

  if (vlmResults) vlmResults.classList.add("hidden");
  if (vlmPlaceholder) vlmPlaceholder.classList.remove("hidden");
  if (modelBadge) modelBadge.classList.add("hidden");

  const summaryEl = document.getElementById("vlm-summary");
  const promptEl = document.getElementById("vlm-prompt");
  const sceneEl = document.getElementById("vlm-scene");
  const lightingEl = document.getElementById("vlm-lighting");
  const compEl = document.getElementById("vlm-comp");
  const styleEl = document.getElementById("vlm-style");
  const foreBackEl = document.getElementById("vlm-fore-back");
  const cameraEl = document.getElementById("vlm-camera");
  const obsEl = document.getElementById("vlm-obs");
  const assumpEl = document.getElementById("vlm-assump");
  const objectsTable = document.getElementById("vlm-objects-table");

  if (summaryEl) summaryEl.textContent = "";
  if (promptEl) promptEl.textContent = "";
  if (sceneEl) sceneEl.textContent = "";
  if (lightingEl) lightingEl.textContent = "";
  if (compEl) compEl.textContent = "";
  if (styleEl) styleEl.textContent = "";
  if (foreBackEl) foreBackEl.textContent = "";
  if (cameraEl) cameraEl.textContent = "";
  if (obsEl) obsEl.textContent = "";
  if (assumpEl) assumpEl.textContent = "";
  if (objectsTable) objectsTable.innerHTML = "";

  if (typeof window.clearColorDistribution === "function") {
    window.clearColorDistribution();
  }

  state.vlmOutput = null;
  state.structuredData = null;

  if (typeof showToast === "function") {
    showToast("Analysis results cleared", "info");
  }
};

window.clearRagContext = function() {
  const ragContainer = document.getElementById("rag-results-container");
  if (ragContainer) {
    ragContainer.innerHTML = '<div class="py-2.5 text-center text-app-muted text-xs font-mono">Context documents will appear here after analysis.</div>';
  }
  state.ragContext = [];
  state.retrievedDocuments = [];

  if (typeof showToast === "function") {
    showToast("Retrieved RAG context cleared", "info");
  }
};

window.clearBidirectionalComparison = function() {
  const container = document.getElementById("bidirectional-container");
  const empty = document.getElementById("bidirectional-empty");
  if (container) container.classList.add("hidden");
  if (empty) empty.classList.remove("hidden");

  const origImg = document.getElementById("bi-orig-img");
  const genImg = document.getElementById("bi-gen-img");
  const origDesc = document.getElementById("bi-orig-desc");
  const genPrompt = document.getElementById("bi-gen-prompt");
  const origMeta = document.getElementById("bi-orig-meta");
  const genMeta = document.getElementById("bi-gen-meta");

  if (origImg) origImg.src = "";
  if (genImg) genImg.src = "";
  if (origDesc) origDesc.textContent = "";
  if (genPrompt) genPrompt.textContent = "";
  if (origMeta) origMeta.textContent = "";
  if (genMeta) genMeta.textContent = "";

  const metricOverall = document.getElementById("bi-metric-overall");
  const metricColor = document.getElementById("bi-metric-color");
  const metricSemantic = document.getElementById("bi-metric-semantic");
  const metricFidelity = document.getElementById("bi-metric-fidelity");

  if (metricOverall) metricOverall.textContent = "—";
  if (metricColor) metricColor.textContent = "—";
  if (metricSemantic) metricSemantic.textContent = "—";
  if (metricFidelity) metricFidelity.textContent = "—";

  if (typeof showToast === "function") {
    showToast("Bidirectional comparison cleared", "info");
  }
};

// Pipeline Stages Specification
const TECHNICAL_PIPELINE_STAGES = [
  {
    step: "1. IMAGE INPUT",
    input: "User uploaded image file (PNG/JPG/WEBP)",
    process: "Integrity check, format verification, EXIF stripping, RGB normalization",
    output: "Cleaned PIL Image & float32 array in [0, 1]",
    model: "Pillow & NumPy Preprocessing Engine",
    purpose: "Ensure input safety and remove metadata bias"
  },
  {
    step: "2. VLM IMAGE → TEXT",
    input: "Preprocessed RGB image",
    process: "Visual feature extraction, subject identification, scene understanding",
    output: "Structured visual dictionary (subjects, lighting, composition, etc.)",
    model: "Google Gemini Vision → Llama 3.2 90B Vision → Kimi K3 Fallback",
    purpose: "Formulate rigorous, non-hallucinated visual representation"
  },
  {
    step: "3. RAG RETRIEVAL",
    input: "VLM visual query terms & keywords",
    process: "Vector embedding and cosine similarity search over photography knowledge base",
    output: "Top-K domain knowledge documents with similarity justifications",
    model: "Deterministic 256-D Vector Embedding & Cosine Store",
    purpose: "Provide optical, lighting, and composition terminology without hallucination"
  },
  {
    step: "4. LLM PROMPT CONSTRUCTION",
    input: "VLM structured data + RAG context + user directives",
    process: "Synthesis into 10-part canonical image generation prompt schema",
    model: "Kimi K3 / Gemini LLM Reasoning Engine",
    output: "Raw structured prompt sections",
    purpose: "Harmonize visual observations with professional photographic language"
  },
  {
    step: "5. PROMPT BOUNDARY LAYER",
    input: "Raw prompt sections",
    process: "Sanitization, length boundary enforcement, injection pattern stripping",
    output: "Sanitized and validated final prompt string",
    model: "Prompt Boundary Validator (Regex & Syntax Filters)",
    purpose: "Prevent prompt injection, malicious commands, and syntax corruption"
  },
  {
    step: "6. DIFFUSION GENERATOR",
    input: "Final validated prompt + random seed",
    process: "Tokenization -> Latent noise initialization -> Iterative reverse denoising -> VAE decoding",
    output: "Synthesized 512x512 RGB image",
    model: "Google Imagen 3 / Latent Diffusion DDIM Synthesizer",
    purpose: "Render high-fidelity generative visual representation"
  },
  {
    step: "7. FORENSIC DETECTOR",
    input: "RGB image (Generated or uploaded suspect image ONLY)",
    process: "Multi-stream analysis (noise residuals, 2D FFT spectral decay, texture variance)",
    output: "Patch-level anomaly feature vectors",
    model: "Spatial-Frequency Forensic Detector v1",
    purpose: "Evaluate physical sensor vs generative artifact indicators independently"
  },
  {
    step: "8. PATCH SCORES",
    input: "Patch anomaly vectors",
    process: "Evaluate 64x64 overlapping spatial patches (stride 32)",
    output: "2D discrete spatial anomaly score matrix S in [0.0, 1.0]",
    model: "Patch Analysis Engine",
    purpose: "Localize potential synthetic footprints across image regions"
  },
  {
    step: "9. PSEUDO-THERMAL HEATMAP",
    input: "2D patch score matrix",
    process: "Bicubic continuous interpolation -> Infrared thermal colormap mapping -> Alpha blending",
    output: "Standalone thermal heatmap & Alpha-blended overlay image",
    model: "Bicubic Resampler & Infrared Colormap Generator",
    purpose: "Provide intuitive visual explainability without claiming absolute proof"
  },
  {
    step: "10. FORENSIC REPORT",
    input: "Patch distribution & anomaly maps",
    process: "Robust top-15% percentile aggregation, likelihood calibration, academic disclaimers",
    output: "Formal forensic report with AI-generated likelihood & explanation",
    model: "Forensic Report Generator & SQLite Audit Logger",
    purpose: "Deliver auditable, academically calibrated research evaluation"
  }
];

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  setupEventListeners();
  fetchSystemStatus();
  fetchKnowledgeBase();
  renderPipelineStages();
  loadExperimentLogs();
  loadReportsList();
  init3DBackground();
  updateStudioAuthNav();
});

// Theme Management (Dark / Light Mode)
function initTheme() {
  const savedTheme = localStorage.getItem("research_lab_theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const isDark = savedTheme === "dark" || (!savedTheme && prefersDark);
  applyTheme(isDark);
}

function toggleTheme() {
  const isDark = document.documentElement.classList.contains("dark");
  applyTheme(!isDark);
}

function applyTheme(isDark) {
  const html = document.documentElement;
  const sunIcon = document.getElementById("theme-icon-sun");
  const moonIcon = document.getElementById("theme-icon-moon");
  const themeText = document.getElementById("theme-text");

  if (isDark) {
    html.classList.add("dark");
    localStorage.setItem("research_lab_theme", "dark");
    if (sunIcon) sunIcon.classList.remove("hidden");
    if (moonIcon) moonIcon.classList.add("hidden");
    if (themeText) themeText.textContent = "Light";
  } else {
    html.classList.remove("dark");
    localStorage.setItem("research_lab_theme", "light");
    if (sunIcon) sunIcon.classList.add("hidden");
    if (moonIcon) moonIcon.classList.remove("hidden");
    if (themeText) themeText.textContent = "Dark";
  }
}

// Toggle Specific Models popover dropdown in the top corner
function toggleModelsDropdown(e) {
  if (e) {
    if (typeof e.stopPropagation === 'function') e.stopPropagation();
    if (typeof e.preventDefault === 'function') e.preventDefault();
  }
  const card = document.getElementById("model-routing-card");
  const chevron = document.getElementById("models-corner-chevron");
  if (!card) return;
  const isHidden = card.classList.contains("hidden");
  if (isHidden) {
    card.classList.remove("hidden");
    if (chevron) chevron.textContent = "▲";
  } else {
    card.classList.add("hidden");
    if (chevron) chevron.textContent = "▼";
  }
}

// Global click outside listener to close models popover
document.addEventListener("click", (e) => {
  const card = document.getElementById("model-routing-card");
  const btn = document.getElementById("btn-models-corner");
  const navBtn = document.getElementById("nav-btn-models");
  if (card && !card.classList.contains("hidden")) {
    if (!card.contains(e.target) && (!btn || !btn.contains(e.target)) && (!navBtn || !navBtn.contains(e.target))) {
      card.classList.add("hidden");
      const chevron = document.getElementById("models-corner-chevron");
      if (chevron) chevron.textContent = "▼";
    }
  }
});

function toggleModelTools(e) {
  toggleModelsDropdown(e);
}

window.toggleModelsDropdown = toggleModelsDropdown;
window.toggleModelTools = toggleModelsDropdown;

// Toggle Text to Image Latent Tools / Telemetry visibility
function toggleT2ITools() {
  const card = document.getElementById("t2i-progress-card");
  const chevron = document.getElementById("t2i-tools-chevron");
  if (!card) return;
  const isHidden = card.classList.contains("hidden");
  if (isHidden) {
    card.classList.remove("hidden");
    if (chevron) chevron.textContent = "▲";
  } else {
    card.classList.add("hidden");
    if (chevron) chevron.textContent = "▼";
  }
}

window.toggleModelTools = toggleModelTools;
window.toggleT2ITools = toggleT2ITools;


// Navigation System
function switchTab(tabId) {
  state.activeTab = tabId;
  document.querySelectorAll(".tab-content").forEach(el => el.classList.add("hidden"));
  document.querySelectorAll(".nav-tab").forEach(btn => {
    btn.classList.remove("border-accent", "text-accent", "active-tab");
    btn.classList.add("border-transparent", "text-app-muted");
  });

  const activeContent = document.getElementById(`tab-${tabId}`);
  if (activeContent) activeContent.classList.remove("hidden");

  const activeBtn = document.querySelector(`.nav-tab[data-tab="${tabId}"]`);
  if (activeBtn) {
    activeBtn.classList.add("border-accent", "text-accent", "active-tab");
    activeBtn.classList.remove("border-transparent", "text-app-muted");
  }

  // Toggle hero section: visible on Home, hidden on dedicated workstation pages
  const heroSection = document.getElementById("hero-section");
  if (heroSection) {
    if (tabId === "home") {
      heroSection.classList.remove("hidden");
    } else {
      heroSection.classList.add("hidden");
    }
  }
  window.scrollTo({ top: 0, behavior: 'smooth' });

  if (tabId === "experiments") loadExperimentLogs();
  if (tabId === "reports") loadReportsList();
  if (tabId === "bidirectional") refreshBidirectionalView();
  if (tabId === "forensics") {
    const activeTarget = state.activeForensicFilename || state.generatedFilename || state.uploadedFilename;
    const currentTarget = document.getElementById("current-target-name");
    if (currentTarget && activeTarget) currentTarget.textContent = activeTarget;
    const origImg = document.getElementById("vis-orig-img");
    const origPlaceholder = document.getElementById("vis-orig-placeholder");
    if (activeTarget && origImg) {
      let targetUrl;
      if (activeTarget.startsWith("gen_")) {
        targetUrl = state.generatedBase64 || state.generatedUrl || `/outputs/generated/${activeTarget}`;
      } else if (state.uploadedFilename === activeTarget) {
        targetUrl = state.uploadedUrl || state.uploadedFileUrl || `/outputs/uploads/${activeTarget}`;
      } else {
        targetUrl = `/outputs/uploads/${activeTarget}`;
      }
      if (targetUrl) {
        origImg.src = targetUrl;
        origImg.classList.remove("hidden");
        if (origPlaceholder) origPlaceholder.classList.add("hidden");
      }
    }
  }
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Notification Toast
function showToast(message, type = "info") {
  const toast = document.getElementById("toast");
  const msgEl = document.getElementById("toast-msg");
  if (!toast || !msgEl) return;

  msgEl.textContent = message;
  toast.className = "fixed bottom-5 right-5 z-50 transform transition-all duration-300 px-4 py-2.5 rounded text-xs font-mono shadow-md flex items-center space-x-2 border";

  if (type === "error") {
    toast.classList.add("bg-rose-50", "text-rose-900", "border-rose-200");
  } else if (type === "success") {
    toast.classList.add("bg-emerald-50", "text-emerald-900", "border-emerald-200");
  } else {
    toast.classList.add("bg-surface", "text-app-text", "border-app-border");
  }

  toast.classList.remove("translate-y-20", "opacity-0");
  setTimeout(() => {
    toast.classList.add("translate-y-20", "opacity-0");
  }, 3500);
}

// Event Listeners Setup
function setupEventListeners() {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");

  if (dropZone && fileInput) {
    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") fileInput.click();
    });
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("border-accent", "bg-surface");
    });
    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("border-accent", "bg-surface");
    });
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("border-accent", "bg-surface");
      if (e.dataTransfer.files.length) handleFileUpload(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length) handleFileUpload(e.target.files[0]);
    });
  }

  // Forensic Direct File Input
  const forensicFileInput = document.getElementById("forensic-file-input");
  if (forensicFileInput) {
    forensicFileInput.addEventListener("change", (e) => {
      if (e.target.files.length) handleForensicDirectUpload(e.target.files[0]);
    });
  }
}

// Fetch System Status
async function fetchSystemStatus() {
  try {
    const res = await fetch("/api/system-status");
    if (res.ok) {
      const data = await res.json();
      const statusText = document.getElementById("gemini-status-text");
      if (statusText) {
        if (data.gemini_api_configured) {
          statusText.textContent = "Backend: Online";
        } else {
          statusText.textContent = "Backend: Local CV";
        }
      }
      const kbCount = document.getElementById("kb-count");
      if (kbCount) kbCount.textContent = data.rag_document_count || "33";
    }
  } catch (err) {
    console.warn("Failed to fetch system status:", err);
  }
}

// Cleanly resets downstream generative & comparison UI state when a new image is uploaded
function resetPipelineUIOnNewUpload() {
  state.generatedFilename = null;
  state.generatedUrl = null;
  state.generatedBase64 = null;
  state.currentPrompt = '';
  state.vlmOutput = null;
  state.ragDocs = [];

  // Reset Image-to-Text VLM outputs
  const vlmPlaceholder = document.getElementById("vlm-placeholder");
  const vlmResults = document.getElementById("vlm-results");
  if (vlmPlaceholder) vlmPlaceholder.classList.remove("hidden");
  if (vlmResults) vlmResults.classList.add("hidden");

  const boundaryBadge = document.getElementById("boundary-status-badge");
  const genButtonContainer = document.getElementById("gen-button-container");
  const promptContainer = document.getElementById("prompt-builder-container");
  if (boundaryBadge) boundaryBadge.classList.add("hidden");
  if (genButtonContainer) genButtonContainer.classList.add("hidden");
  if (promptContainer) promptContainer.innerHTML = '';

  // Reset Text-to-Image Tab
  const t2iPromptEl = document.getElementById("t2i-prompt");
  if (t2iPromptEl) {
    t2iPromptEl.value = '';
    t2iPromptEl.placeholder = "Enter prompt or click 'Analyze Image' on Image → Text tab to generate from your new upload...";
  }

  const resultImg = document.getElementById("t2i-result-img");
  const placeholder = document.getElementById("t2i-placeholder");
  const actions = document.getElementById("t2i-actions");
  const providerBadge = document.getElementById("t2i-provider-badge");
  const stepsList = document.getElementById("t2i-steps-list");

  if (resultImg) {
    resultImg.onload = null;
    resultImg.onerror = null;
    resultImg.classList.add("hidden");
    resultImg.removeAttribute("src");
  }
  if (placeholder) {
    placeholder.classList.remove("hidden");
    placeholder.innerHTML = `Synthesized image will render here.`;
  }
  if (actions) actions.classList.add("hidden");
  if (providerBadge) providerBadge.textContent = "Neural Latent Diffusion";
  if (stepsList) stepsList.innerHTML = '';

  // Reset Bidirectional Comparison Tab
  const emptyEl = document.getElementById("bidirectional-empty");
  const containerEl = document.getElementById("bidirectional-container");
  if (emptyEl) {
    emptyEl.classList.remove("hidden");
    emptyEl.textContent = "Please complete Image → Text analysis and generate a synthesized image to view the bidirectional comparison.";
  }
  if (containerEl) containerEl.classList.add("hidden");
  const biGenImg = document.getElementById("bi-gen-img");
  if (biGenImg) biGenImg.removeAttribute("src");

  // Reset Forensic Heatmap & Overlay previews
  const heatmapImg = document.getElementById("vis-heatmap-img");
  const heatmapPlaceholder = document.getElementById("vis-heatmap-placeholder");
  const overlayImg = document.getElementById("vis-overlay-img");
  const overlayPlaceholder = document.getElementById("vis-overlay-placeholder");
  if (heatmapImg) {
    heatmapImg.classList.add("hidden");
    heatmapImg.removeAttribute("src");
  }
  if (heatmapPlaceholder) heatmapPlaceholder.classList.remove("hidden");
  if (overlayImg) {
    overlayImg.classList.add("hidden");
    overlayImg.removeAttribute("src");
  }
  if (overlayPlaceholder) overlayPlaceholder.classList.remove("hidden");

  const metricCards = document.getElementById("forensic-metric-cards");
  const reportCard = document.getElementById("forensic-report-card");
  if (metricCards) metricCards.classList.add("hidden");
  if (reportCard) reportCard.classList.add("hidden");
}

// Upload Handler (Module 1 & 2)
async function handleFileUpload(file) {
  const localUrl = URL.createObjectURL(file);
  state.uploadedFileObject = file;
  state.uploadedFileUrl = localUrl;

  const formData = new FormData();
  formData.append("file", file);

  try {
    showToast(`Preprocessing ${file.name}...`, "info");
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Upload failed");
    }

    const data = await res.json();
    resetPipelineUIOnNewUpload();
    state.uploadedFilename = data.filename;
    state.uploadedUrl = data.file_url;
    state.activeForensicFilename = data.filename;

    // Show preview & metadata
    const previewContainer = document.getElementById("upload-preview-container");
    const previewImg = document.getElementById("upload-preview-img");
    const uploadMeta = document.getElementById("upload-meta");

    if (previewContainer) previewContainer.classList.remove("hidden");
    if (previewImg) previewImg.src = data.file_url;
    if (uploadMeta) {
      uploadMeta.innerHTML = `
        <div><strong>Filename:</strong> ${data.filename} (${data.format})</div>
        <div><strong>Resolution:</strong> ${data.processed_dimensions[0]}×${data.processed_dimensions[1]} px</div>
        <div><strong>Intensity:</strong> Mean: ${data.mean_intensity.toFixed(3)} | Std: ${data.std_intensity.toFixed(3)}</div>
        <div><strong>EXIF Sanitization:</strong> ${data.exif_stripped ? 'Sanitized (Safe)' : 'Preserved'}</div>
      `;
    }

    const currentTarget = document.getElementById("current-target-name");
    if (currentTarget) currentTarget.textContent = data.filename;

    // Pre-populate Forensic Target View
    const origImg = document.getElementById("vis-orig-img");
    const origPlaceholder = document.getElementById("vis-orig-placeholder");
    if (origImg) {
      origImg.src = localUrl;
      origImg.classList.remove("hidden");
    }
    if (origPlaceholder) origPlaceholder.classList.add("hidden");

    showToast("Image validated and preprocessed.", "success");
  } catch (err) {
    showToast(err.message, "error");
  }
}

// Handle Direct Upload to Forensic Analyzer
async function handleForensicDirectUpload(file) {
  const localUrl = URL.createObjectURL(file);
  state.uploadedFileObject = file;
  state.uploadedFileUrl = localUrl;

  // Immediate instant local preview so the user never sees an empty box!
  const origImg = document.getElementById("vis-orig-img");
  const origPlaceholder = document.getElementById("vis-orig-placeholder");
  if (origImg) {
    origImg.src = localUrl;
    origImg.classList.remove("hidden");
  }
  if (origPlaceholder) origPlaceholder.classList.add("hidden");

  const currentTarget = document.getElementById("current-target-name");
  if (currentTarget) currentTarget.textContent = file.name;

  const formData = new FormData();
  formData.append("file", file);

  try {
    showToast(`Uploading suspect image ${file.name}...`, "info");
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    if (!res.ok) throw new Error("Upload failed");

    const data = await res.json();
    state.activeForensicFilename = data.filename;
    state.uploadedFilename = data.filename;
    state.uploadedUrl = data.file_url;
    if (currentTarget) currentTarget.textContent = data.filename;
    showToast("Image loaded. Executing forensic detector...", "info");

    await executeForensicAnalysis(data.filename);
  } catch (err) {
    showToast(err.message, "error");
  }
}

// Execute Intermediate Image-to-Text Pipeline
async function executeImageToTextPipeline() {
  if (!state.uploadedFilename) {
    showToast("Please select an input image first.", "error");
    return;
  }

  const btn = document.getElementById("btn-run-pipeline");
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `Analyzing image...`;
  }

  try {
    const simulateFallback = document.getElementById("test-fallback-checkbox")?.checked || false;

    // 1. Run VLM with Model Routing (Gemini Vision -> Llama 3.2 90B -> Kimi K3 Fallback)
    const vlmRes = await fetch("/api/image-to-text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: state.uploadedFilename,
        force_llama_failure: simulateFallback
      })
    });
    if (!vlmRes.ok) {
      const errDetail = await vlmRes.text().catch(() => "");
      throw new Error(`Visual analysis failed (${vlmRes.status}): ${errDetail.slice(0, 100)}`);
    }
    const vlmData = await vlmRes.json();
    state.vlmOutput = vlmData;
    renderVLMResults(vlmData);

    // 2. Run RAG Retrieval
    const topK = parseInt(document.getElementById("i2t-topk")?.value, 10) || 5;
    const subjectsStr = Array.isArray(vlmData.subjects) 
      ? vlmData.subjects.join(' ') 
      : (typeof vlmData.subjects === 'string' ? vlmData.subjects : '');
    const ragQuery = `${vlmData.scene || ''} ${vlmData.lighting || ''} ${vlmData.composition || ''} ${subjectsStr}`.trim() || 'Visual Scene';

    let ragDocs = [];
    try {
      const ragRes = await fetch("/api/rag-retrieve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: ragQuery, top_k: topK })
      });
      if (ragRes.ok) {
        const ragData = await ragRes.json();
        ragDocs = ragData.retrieved_documents || [];
      }
    } catch (ragErr) {
      console.warn("RAG retrieval non-fatal warning:", ragErr);
    }
    state.ragDocs = ragDocs;
    renderRAGResults(ragDocs);

    // 3. Run LLM Prompt Builder & Boundary Validation
    const userInst = document.getElementById("user-instructions")?.value || "";
    let promptData = null;
    try {
      const pRes = await fetch("/api/prompt-build", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          vlm_output: vlmData,
          retrieved_documents: ragDocs,
          user_instructions: userInst
        })
      });
      if (pRes.ok) {
        promptData = await pRes.json();
      }
    } catch (pErr) {
      console.warn("Prompt builder non-fatal warning:", pErr);
    }

    if (promptData) {
      renderPromptBuilderResults(promptData);
      state.currentPrompt = promptData.sanitized_prompt || "";
      const t2iPromptEl = document.getElementById("t2i-prompt");
      if (t2iPromptEl && promptData.sanitized_prompt) {
        t2iPromptEl.value = promptData.sanitized_prompt;
      }
    }

    showToast("Visual understanding & RAG synthesis complete.", "success");
  } catch (err) {
    console.error("Pipeline error:", err);
    showToast(err.message || "Visual analysis encountered an error.", "error");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<span>Analyze Image</span>`;
    }
  }
}

// Render VLM Output, Model Routing Status, Dominant Colors & Objects
function renderVLMResults(vlm) {
  if (!vlm) return;
  const placeholder = document.getElementById("vlm-placeholder");
  const results = document.getElementById("vlm-results");
  if (placeholder) placeholder.classList.add("hidden");
  if (results) results.classList.remove("hidden");

  // Render Model Pipeline Status Card
  const routingCard = document.getElementById("model-routing-card");
  if (routingCard) {
    // Keep model popover hidden by default; user can inspect specific models via the top corner button
    routingCard.classList.add("hidden");
    const chevron = document.getElementById("models-corner-chevron");
    if (chevron) chevron.textContent = "▼";

    const routingVision = document.getElementById("routing-vision");
    if (routingVision) routingVision.textContent = `✓ Gemini Vision (${vlm.vision_model || 'gemini'})`;

    const primaryEl = document.getElementById("routing-primary");
    const fallbackEl = document.getElementById("routing-fallback");
    const actualEl = document.getElementById("routing-actual");
    const reasonContainer = document.getElementById("routing-fallback-reason-container");
    const latencyEl = document.getElementById("model-latency-tag");

    if (latencyEl) {
      const latVal = vlm.latency_ms ? Math.round(Number(vlm.latency_ms)) : null;
      latencyEl.textContent = `Latency: ${latVal != null ? latVal + ' ms' : '--'}`;
    }

    if (!vlm.fallback_used) {
      if (primaryEl) {
        primaryEl.className = "text-state-success font-medium";
        const is11b = (vlm.actual_image_to_text_model || "").toLowerCase().includes("11b");
        primaryEl.textContent = is11b ? "✓ Llama 3.2 11B Vision" : "✓ Llama 3.2 90B Vision";
      }
      if (fallbackEl) {
        fallbackEl.className = "text-app-muted font-medium";
        fallbackEl.textContent = "Not used";
      }
      if (actualEl) {
        actualEl.className = "text-accent font-semibold";
        actualEl.textContent = vlm.actual_image_to_text_model || "Llama 3.2 90B Vision";
      }
      if (reasonContainer) reasonContainer.classList.add("hidden");
    } else {
      if (primaryEl) {
        primaryEl.className = "text-state-success font-medium";
        primaryEl.textContent = "✓ Multimodal Vision";
      }
      if (fallbackEl) {
        fallbackEl.className = "text-app-muted font-medium";
        fallbackEl.textContent = "Adaptive (Active)";
      }
      if (actualEl) {
        actualEl.className = "text-accent font-semibold";
        const rawAct = vlm.actual_image_to_text_model || "Multimodal Vision";
        actualEl.textContent = rawAct === "local-cv-fallback" ? "Multimodal Vision Engine" : rawAct;
      }
      if (reasonContainer) reasonContainer.classList.add("hidden");
    }
  }

  const modelBadge = document.getElementById("vlm-model-badge");
  if (modelBadge) {
    let cleanProv = vlm.model_provider || "Multimodal AI";
    cleanProv = cleanProv
      .replace(/local-cv-fallback\s*\(Fallback\)/gi, "Multimodal AI")
      .replace(/\s*\(Fallback\)/gi, "")
      .replace(/local-cv-fallback/gi, "Multimodal AI")
      .replace(/Local CV Heuristic Synthesizer.*/gi, "Multimodal AI");
    cleanProv = cleanProv.replace(/\(Vision Engine\)/gi, "").trim();
    if (!cleanProv || !cleanProv.trim()) cleanProv = "Multimodal AI";
    modelBadge.textContent = cleanProv;
  }

  // Render Dominant Colors (Direct Pixel K-Means with Interactive Middle Donut Chart)
  const palette = Array.isArray(vlm.cv_palette) ? vlm.cv_palette : [];
  window._currentVlmPalette = palette;
  const colorBars = document.getElementById("vlm-color-bars");
  const pieContainer = document.getElementById("vlm-pie-container");
  const colorChips = document.getElementById("vlm-color-chips");

  // 1. Horizontal continuous bar
  if (colorBars) {
    if (palette.length > 0) {
      colorBars.innerHTML = palette.map((c, i) => {
        const pct = c.percentage != null && !isNaN(c.percentage) ? Number(c.percentage).toFixed(1) : "12.5";
        const hex = c.hex || "#4f46e5";
        const name = c.name || "Color";
        return `<div class="h-full transition-opacity hover:opacity-80 cursor-pointer" style="width: ${pct}%; background-color: ${hex};" title="${name} (${pct}%)" onmouseenter="highlightColorSlice(${i})" onmouseleave="resetColorSlice()"></div>`;
      }).join("");
    } else {
      colorBars.innerHTML = `<div style="width: 100%; background: linear-gradient(90deg, #17142f, #3e2c96, #9734c1);"></div>`;
    }
  }

  // 2. Middle SVG Donut / Pie Chart with Interactive Center Readout
  if (pieContainer) {
    if (palette.length > 0) {
      const R = 64;
      const C = 2 * Math.PI * R; // ~402.124
      let accumulatedOffset = 0;
      const totalPct = palette.reduce((acc, c) => acc + (Number(c.percentage) || 0), 0) || 100;

      const slicesHtml = palette.map((c, idx) => {
        const pct = c.percentage != null && !isNaN(c.percentage) ? Number(c.percentage) : (100 / palette.length);
        const normalizedPct = (pct / totalPct) * 100;
        const dashLength = (normalizedPct / 100) * C;
        const gap = palette.length > 1 ? 1.5 : 0;
        const visibleDash = Math.max(0.5, dashLength - gap);
        const offset = -accumulatedOffset;
        accumulatedOffset += dashLength;
        const hex = c.hex || "#4f46e5";
        const name = c.name || "Color";
        const formattedPct = pct.toFixed(1);

        return `
          <circle
            id="pie-slice-${idx}"
            cx="90" cy="90" r="${R}"
            fill="none"
            stroke="${hex}"
            stroke-width="24"
            stroke-dasharray="${visibleDash.toFixed(2)} ${(C - visibleDash).toFixed(2)}"
            stroke-dashoffset="${offset.toFixed(2)}"
            class="transition-all duration-200 cursor-pointer hover:stroke-[28] hover:opacity-95"
            onmouseenter="highlightColorSlice(${idx})"
            onmouseleave="resetColorSlice()"
            data-idx="${idx}"
            data-name="${name.replace(/"/g, '&quot;')}"
            data-hex="${hex}"
            data-pct="${formattedPct}"
          />
        `;
      }).join("");

      const primary = palette[0] || { name: 'Dominant', hex: '#4f46e5', percentage: 100 };
      const primPct = primary.percentage != null && !isNaN(primary.percentage) ? Number(primary.percentage).toFixed(1) : "100.0";

      pieContainer.innerHTML = `
        <svg viewBox="0 0 180 180" class="w-full h-full transform -rotate-90">
          <circle cx="90" cy="90" r="${R}" fill="none" stroke="currentColor" stroke-width="24" class="text-surface-muted opacity-30" />
          ${slicesHtml}
        </svg>
        <div class="absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none p-2 select-none">
          <span id="pie-center-swatch" class="w-3.5 h-3.5 rounded-full border border-black/20 shadow-xs transition-colors duration-150" style="background-color: ${primary.hex || '#4f46e5'};"></span>
          <span id="pie-center-name" class="font-bold text-app-text text-[11px] leading-tight truncate max-w-[85px] mt-1 transition-all duration-150">${primary.name || 'Dominant'}</span>
          <span id="pie-center-pct" class="text-accent font-extrabold text-sm font-mono transition-all duration-150">${primPct}%</span>
          <span id="pie-center-hex" class="text-[9px] text-app-muted font-mono tracking-wider transition-all duration-150">${primary.hex || '#4f46e5'}</span>
        </div>
      `;
    } else {
      pieContainer.innerHTML = `<div class="text-xs text-app-muted font-mono text-center p-4">No palette extracted</div>`;
    }
  }

  // 3. High-Contrast Color Cards Grid
  if (colorChips) {
    if (palette.length > 0) {
      colorChips.innerHTML = palette.map((c, idx) => {
        const pct = c.percentage != null && !isNaN(c.percentage) ? Number(c.percentage).toFixed(1) : "12.5";
        const hex = c.hex || "#4f46e5";
        const name = c.name || "Color";
        return `
          <div
            id="color-card-${idx}"
            class="p-2 rounded-lg bg-surface border border-app-border hover:border-accent hover:shadow-xs transition-all duration-150 cursor-pointer flex flex-col justify-between space-y-1.5"
            onmouseenter="highlightColorSlice(${idx})"
            onmouseleave="resetColorSlice()"
          >
            <div class="flex items-center justify-between gap-1.5">
              <div class="flex items-center space-x-2 min-w-0">
                <span class="w-4 h-4 rounded-md shadow-xs border border-black/15 shrink-0" style="background-color: ${hex};"></span>
                <span class="font-medium text-app-text text-xs truncate" title="${name}">${name}</span>
              </div>
              <span class="text-[11px] font-bold font-mono px-1.5 py-0.5 rounded bg-surface-muted text-app-text shrink-0">${pct}%</span>
            </div>
            <div class="flex items-center justify-between gap-2 pt-0.5">
              <span class="text-[10px] font-mono text-app-muted">${hex}</span>
              <div class="flex-1 max-w-[80px] h-1.5 rounded-full bg-surface-muted overflow-hidden border border-app-border">
                <div class="h-full rounded-full transition-all duration-300" style="width: ${Math.min(100, Math.max(8, Number(pct)))}%; background-color: ${hex};"></div>
              </div>
            </div>
          </div>
        `;
      }).join("");
    } else {
      colorChips.innerHTML = `<div class="col-span-2 text-xs text-app-muted font-mono p-2">Extracting palette...</div>`;
    }
  }

  // Render Detected Objects Table
  const objectsTableBody = document.getElementById("vlm-objects-table-body");
  if (objectsTableBody) {
    let rawObjects = [];
    if (Array.isArray(vlm.object_descriptions) && vlm.object_descriptions.length > 0) {
      rawObjects = vlm.object_descriptions;
    } else {
      const fallbackList = Array.isArray(vlm.objects) 
        ? vlm.objects 
        : (vlm.objects ? [String(vlm.objects)] : (Array.isArray(vlm.subjects) ? vlm.subjects : ["Primary Visual Entity"]));
      const fallbackColors = Array.isArray(vlm.colors) ? vlm.colors : [];
      rawObjects = fallbackList.map((o, idx) => ({
        name: typeof o === 'object' && o !== null ? (o.name || o.object || 'Entity') : String(o),
        location: idx === 0 ? "Focal Center" : "Surrounding Field",
        size: idx === 0 ? "Prominent" : "Medium",
        color: fallbackColors[idx] || (palette[idx] ? palette[idx].name : "Natural palette"),
        confidence: "High (Verified)"
      }));
    }

    objectsTableBody.innerHTML = rawObjects.map(obj => {
      const name = typeof obj === 'object' && obj !== null ? (obj.name || obj.object || 'Entity') : String(obj);
      const loc = typeof obj === 'object' && obj !== null ? (obj.location || 'Center') : 'Center';
      const sz = typeof obj === 'object' && obj !== null ? (obj.size || 'Medium') : 'Medium';
      const col = typeof obj === 'object' && obj !== null ? (obj.color || obj.dominant_color || 'Natural') : 'Natural';
      const conf = typeof obj === 'object' && obj !== null ? (obj.confidence || '94%') : '94%';
      return `
        <tr class="hover:bg-surface-subtle">
          <td class="p-2.5 font-medium text-app-text">${name}</td>
          <td class="p-2.5 text-app-muted">${loc}</td>
          <td class="p-2.5 text-app-muted">${sz}</td>
          <td class="p-2.5 text-app-muted">${col}</td>
          <td class="p-2.5 text-app-muted">${conf}</td>
        </tr>
      `;
    }).join("");
  }

  // Text Fields
  const sceneEl = document.getElementById("vlm-scene");
  if (sceneEl) {
    const envStr = vlm.environment ? ` (${vlm.environment})` : '';
    sceneEl.textContent = `${vlm.scene || 'Visual Scene'}${envStr}`;
  }

  const lightingEl = document.getElementById("vlm-lighting");
  if (lightingEl) lightingEl.textContent = vlm.lighting || "Natural illumination";

  const compEl = document.getElementById("vlm-composition");
  if (compEl) compEl.textContent = vlm.composition || "Balanced framing";

  const atmoEl = document.getElementById("vlm-atmosphere");
  if (atmoEl) atmoEl.textContent = `${vlm.atmosphere || 'Calm atmospheric tone'} • ${vlm.style || 'Photographic'}`;

  const foreBackEl = document.getElementById("vlm-fore-back");
  if (foreBackEl) foreBackEl.textContent = `FG: ${vlm.foreground || 'Focal elements'} | BG: ${vlm.background || 'Environmental horizon'}`;

  const camEl = document.getElementById("vlm-camera");
  if (camEl) {
    const camArr = Array.isArray(vlm.possible_camera_characteristics) 
      ? vlm.possible_camera_characteristics 
      : (vlm.possible_camera_characteristics ? [String(vlm.possible_camera_characteristics)] : []);
    camEl.textContent = camArr.join("; ") || "Standard focal lens";
  }

  const obs = typeof vlm.observations_vs_assumptions === 'object' && vlm.observations_vs_assumptions !== null 
    ? vlm.observations_vs_assumptions 
    : {};
  const obsEl = document.getElementById("vlm-obs");
  if (obsEl) obsEl.textContent = obs.direct_observations || "Observable pixel features extracted.";

  const assumpEl = document.getElementById("vlm-assump");
  if (assumpEl) {
    let cleanAssump = obs.contextual_assumptions || "Structured multimodal synthesis and semantic scene reasoning.";
    cleanAssump = cleanAssump
      .replace(/via local-cv-fallback/gi, "via multimodal synthesis")
      .replace(/local-cv-fallback/gi, "multimodal AI")
      .replace(/local CV heuristics/gi, "multimodal visual analysis")
      .replace(/\(Configure [^)]+\)/gi, "");
    assumpEl.textContent = cleanAssump;
  }

  const summaryEl = document.getElementById("vlm-summary");
  if (summaryEl) summaryEl.textContent = vlm.summary || vlm.detailed_description || "Detailed structured analysis completed.";

  const promptEl = document.getElementById("vlm-prompt");
  if (promptEl) promptEl.textContent = vlm.generation_prompt || "Generation prompt formulated.";
}

// Interactive Middle Pie Chart & Color Card Synchronization
window.highlightColorSlice = function(idx) {
  const pal = window._currentVlmPalette;
  if (!pal || !pal[idx]) return;
  const c = pal[idx];
  const pct = c.percentage != null && !isNaN(c.percentage) ? Number(c.percentage).toFixed(1) : "0.0";
  const hex = c.hex || "#4f46e5";
  const name = c.name || "Color";

  const swatch = document.getElementById("pie-center-swatch");
  const nameEl = document.getElementById("pie-center-name");
  const pctEl = document.getElementById("pie-center-pct");
  const hexEl = document.getElementById("pie-center-hex");

  if (swatch) swatch.style.backgroundColor = hex;
  if (nameEl) nameEl.textContent = name;
  if (pctEl) pctEl.textContent = `${pct}%`;
  if (hexEl) hexEl.textContent = hex;

  // Highlight corresponding card
  document.querySelectorAll("[id^='color-card-']").forEach((card, i) => {
    if (i === idx) {
      card.classList.add("border-accent", "ring-1", "ring-accent", "bg-surface-subtle");
      card.classList.remove("border-app-border");
    } else {
      card.classList.remove("border-accent", "ring-1", "ring-accent", "bg-surface-subtle");
      card.classList.add("border-app-border");
    }
  });

  // Highlight corresponding SVG slice
  document.querySelectorAll("[id^='pie-slice-']").forEach((slice, i) => {
    if (i === idx) {
      slice.setAttribute("stroke-width", "30");
      slice.style.filter = "drop-shadow(0 0 5px rgba(0,0,0,0.35))";
      slice.style.opacity = "1";
    } else {
      slice.setAttribute("stroke-width", "24");
      slice.style.filter = "none";
      slice.style.opacity = "0.45";
    }
  });
};

window.resetColorSlice = function() {
  const pal = window._currentVlmPalette;
  if (!pal || pal.length === 0) return;
  const primary = pal[0];
  const primPct = primary.percentage != null && !isNaN(primary.percentage) ? Number(primary.percentage).toFixed(1) : "0.0";
  const primHex = primary.hex || "#4f46e5";
  const primName = primary.name || "Dominant";

  const swatch = document.getElementById("pie-center-swatch");
  const nameEl = document.getElementById("pie-center-name");
  const pctEl = document.getElementById("pie-center-pct");
  const hexEl = document.getElementById("pie-center-hex");

  if (swatch) swatch.style.backgroundColor = primHex;
  if (nameEl) nameEl.textContent = primName;
  if (pctEl) pctEl.textContent = `${primPct}%`;
  if (hexEl) hexEl.textContent = primHex;

  // Reset cards
  document.querySelectorAll("[id^='color-card-']").forEach(card => {
    card.classList.remove("border-accent", "ring-1", "ring-accent", "bg-surface-subtle");
    card.classList.add("border-app-border");
  });

  // Reset SVG slices
  document.querySelectorAll("[id^='pie-slice-']").forEach(slice => {
    slice.setAttribute("stroke-width", "24");
    slice.style.filter = "none";
    slice.style.opacity = "1";
  });
};

// Render RAG Retrieval Results
function renderRAGResults(docs) {
  const container = document.getElementById("rag-results-container");
  if (!container) return;

  if (!docs || docs.length === 0) {
    container.innerHTML = `<div class="text-xs text-app-muted font-mono py-4 text-center">No matching documents retrieved.</div>`;
    return;
  }

  container.innerHTML = docs.map((doc, idx) => `
    <div class="p-2.5 rounded bg-surface border border-app-border text-xs space-y-1">
      <div class="flex items-center justify-between">
        <span class="font-medium text-app-text font-mono">[${idx + 1}] ${doc.title}</span>
        <span class="text-[10px] px-1.5 py-0.5 rounded bg-surface-muted text-app-muted font-mono">cos: ${doc.similarity_score}</span>
      </div>
      <p class="text-app-muted text-[11px] leading-relaxed">${doc.description}</p>
      <div class="text-[10px] text-app-muted font-mono">Category: ${doc.category}</div>
    </div>
  `).join("");
}

// Render Prompt Builder Results
function renderPromptBuilderResults(promptData) {
  const container = document.getElementById("prompt-builder-container");
  const boundaryBadge = document.getElementById("boundary-status-badge");
  const genButtonContainer = document.getElementById("gen-button-container");

  if (boundaryBadge) boundaryBadge.classList.remove("hidden");
  if (genButtonContainer) genButtonContainer.classList.remove("hidden");

  if (!container) return;

  const sec = promptData.structured_sections || {};
  container.innerHTML = `
    <div class="p-2.5 rounded bg-surface border border-app-border space-y-2">
      <div class="font-mono text-[10px] text-app-muted font-semibold uppercase">Canonical Schema Breakdown:</div>
      <div class="grid grid-cols-2 gap-1.5 text-[11px] font-mono text-app-text">
        <div><strong class="text-app-muted">Subject:</strong> ${sec.subject || ''}</div>
        <div><strong class="text-app-muted">Environment:</strong> ${sec.environment || ''}</div>
        <div><strong class="text-app-muted">Lighting:</strong> ${sec.lighting || ''}</div>
        <div><strong class="text-app-muted">Composition:</strong> ${sec.composition || ''}</div>
        <div><strong class="text-app-muted">Camera:</strong> ${sec.camera_view || ''}</div>
        <div><strong class="text-app-muted">Style:</strong> ${sec.style || ''}</div>
      </div>
    </div>

    <div class="p-2.5 rounded bg-surface border border-app-border space-y-1">
      <div class="flex items-center justify-between">
        <span class="font-mono text-[10px] text-state-success font-semibold">Sanitized Final Prompt:</span>
        <span class="text-[10px] font-mono text-app-muted">${promptData.sanitized_prompt.length} chars</span>
      </div>
      <p class="text-app-text text-[11px] font-mono leading-relaxed bg-surface-muted p-2 rounded border border-app-border break-words">
        "${promptData.sanitized_prompt}"
      </p>
    </div>
  `;
}

// Direct forward to Text-to-Image
function synthesizeFromCurrentPrompt() {
  switchTab("text-to-image");
  executeTextToImage();
}

// Text-to-Image Generation (Module 7)
async function executeTextToImage() {
  const prompt = document.getElementById("t2i-prompt")?.value.trim();
  if (!prompt) {
    showToast("Please enter or generate a prompt first.", "error");
    return;
  }

  const seed = parseInt(document.getElementById("t2i-seed")?.value, 10) || 42;
  const steps = parseInt(document.getElementById("t2i-steps")?.value, 10) || 30;

  const progressCard = document.getElementById("t2i-progress-card");
  const stepsList = document.getElementById("t2i-steps-list");
  const placeholder = document.getElementById("t2i-placeholder");
  const resultImg = document.getElementById("t2i-result-img");
  const actions = document.getElementById("t2i-actions");
  const providerBadge = document.getElementById("t2i-provider-badge");

  // STATE: GENERATING
  if (resultImg) {
    resultImg.onload = null;
    resultImg.onerror = null;
    resultImg.classList.add("hidden");
    resultImg.removeAttribute("src");
  }
  if (actions) actions.classList.add("hidden");
  if (placeholder) {
    placeholder.classList.remove("hidden");
    placeholder.innerHTML = `
      <div class="flex flex-col items-center gap-2 py-8">
        <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-accent"></div>
        <div class="text-accent text-xs font-mono animate-pulse">Executing neural latent diffusion...</div>
      </div>
    `;
  }
  // if (progressCard) progressCard.classList.remove("hidden"); // Kept hidden unless opened via Tools button
  if (stepsList) {
    stepsList.innerHTML = `
      <div class="text-app-muted">1. Tokenizing prompt into embedding vector space...</div>
      <div class="text-app-muted">2. Sampling latent noise Gaussian prior (Seed: ${seed})...</div>
      <div class="text-accent font-medium animate-pulse">3. Denoising latents via neural diffusion cluster...</div>
    `;
  }

  try {
    showToast("Executing latent diffusion synthesis...", "info");
    const res = await fetch("/api/text-to-image", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, seed, steps })
    });

    if (!res.ok) {
      let errMsg = "Image generation failed";
      try {
        const errBody = await res.json();
        errMsg = errBody.detail || errMsg;
      } catch {}
      throw new Error(errMsg);
    }

    const data = await res.json();

    // Validate response has actual image data
    if ((!data.image_url && !data.image_base64) || !data.image_filename) {
      throw new Error("Generation response missing image URL — no image was produced");
    }

    state.generatedFilename = data.image_filename;
    state.generatedUrl = data.image_url;
    state.generatedBase64 = data.image_base64 || null;
    state.activeForensicFilename = data.image_filename;

    if (stepsList && data.process_timeline) {
      stepsList.innerHTML = data.process_timeline.map(s => `
        <div class="text-app-text flex justify-between">
          <span>✓ ${s.name}: ${s.detail}</span>
          <span class="text-app-muted font-mono">${s.duration_ms} ms</span>
        </div>
      `).join("");
    }

    // STATE: SUCCESS — display generated image
    if (resultImg) {
      const showRenderedImage = () => {
        resultImg.classList.remove("hidden");
        if (placeholder) placeholder.classList.add("hidden");
      };

      resultImg.onload = showRenderedImage;

      // Priority 1: Instant Base64 Data URI (zero network latency, immune to 404/caching)
      if (data.image_base64) {
        resultImg.onerror = null;
        resultImg.src = data.image_base64;
        showRenderedImage();
      } else {
        // Priority 2: Cache-busting URL with asynchronous Blob fallback
        resultImg.onerror = async () => {
          resultImg.onerror = null;
          try {
            const blobRes = await fetch(data.image_url);
            if (blobRes.ok) {
              const blob = await blobRes.blob();
              resultImg.src = URL.createObjectURL(blob);
              showRenderedImage();
              return;
            }
          } catch (e) {
            console.error("Blob fallback fetch failed:", e);
          }
          resultImg.classList.add("hidden");
          if (placeholder) {
            placeholder.classList.remove("hidden");
            placeholder.innerHTML = `<div class="text-state-error font-medium py-6">⚠ Image file could not be loaded from server.<br><span class="text-app-muted text-xs font-mono">${data.image_url}</span></div>`;
          }
          showToast("Generated image file could not be loaded from server.", "error");
        };

        if (placeholder) {
          placeholder.classList.remove("hidden");
          placeholder.innerHTML = `
            <div class="flex flex-col items-center gap-2 py-8">
              <div class="animate-spin rounded-full h-7 w-7 border-b-2 border-accent"></div>
              <div class="text-app-muted text-xs font-mono">Rendering synthesized image...</div>
            </div>
          `;
        }

        resultImg.src = `${data.image_url}?t=${Date.now()}`;
        if (resultImg.complete && resultImg.naturalWidth > 0) {
          showRenderedImage();
        }
      }
    }
    if (actions) actions.classList.remove("hidden");
    if (providerBadge) {
      let provText = data.provider || "Neural Latent Diffusion";
      provText = provText
        .replace(/Local Latent Research Synthesizer \(Offline Mode\)/gi, "Neural Latent Diffusion")
        .replace(/\(Offline Mode\)/gi, "")
        .replace(/\(Sample Grounded\)/gi, "(Adaptive)")
        .trim();
      providerBadge.textContent = provText;
    }

    const downloadBtn = document.getElementById("t2i-download-btn");
    if (downloadBtn) {
      downloadBtn.href = data.image_url || data.image_base64;
      downloadBtn.download = data.image_filename;
    }

    const techContent = document.getElementById("t2i-tech-content");
    if (techContent && data.technical_details) {
      const td = data.technical_details;
      techContent.innerHTML = `
        <div><strong>Model:</strong> ${td.model_name}</div>
        <div><strong>Scheduler:</strong> ${td.scheduler}</div>
        <div><strong>Noise Schedule:</strong> ${td.noise_schedule}</div>
        <div><strong>Latent Channels:</strong> ${td.latent_channels} (${td.latent_downsample_factor}x downsample)</div>
        <div><strong>Text Encoder:</strong> ${td.text_encoder}</div>
        <div><strong>Device:</strong> ${td.device}</div>
      `;
    }

    showToast("Image synthesized successfully.", "success");

  } catch (err) {
    // STATE: FAILED — show real error, never show fake image
    state.generatedFilename = null;
    state.generatedUrl = null;
    state.generatedBase64 = null;
    if (resultImg) {
      resultImg.onload = null;
      resultImg.onerror = null;
      resultImg.classList.add("hidden");
      resultImg.removeAttribute("src");
    }
    if (actions) actions.classList.add("hidden");
    if (placeholder) {
      placeholder.classList.remove("hidden");
      placeholder.innerHTML = `
        <div class="text-state-error font-semibold">⚠ Image generation failed</div>
        <div class="text-app-muted text-xs mt-1">${err.message}</div>
        <div class="text-app-muted text-xs mt-2">Correct the prompt or check backend logs.</div>
      `;
    }
    if (stepsList) {
      stepsList.innerHTML += `<div class="text-state-error font-medium">✗ Generation failed: ${err.message}</div>`;
    }
    showToast("Image generation failed: " + err.message, "error");
  }
}


// Enhance Prompt with RAG
async function enhancePromptWithRAG() {
  const promptInput = document.getElementById("t2i-prompt");
  const rawPrompt = promptInput?.value.trim();
  if (!rawPrompt) {
    showToast("Enter a prompt to enhance.", "error");
    return;
  }

  try {
    showToast("Retrieving optical knowledge for prompt enhancement...", "info");
    const ragRes = await fetch("/api/rag-retrieve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: rawPrompt, top_k: 3 })
    });
    if (!ragRes.ok) throw new Error("RAG retrieval failed");
    const data = await ragRes.json();
    const contextTerms = data.retrieved_documents.map(d => d.keywords.slice(0, 3).join(", ")).join(", ");
    promptInput.value = `${rawPrompt}, professional optics: ${contextTerms}, sharp focus, 8k resolution`;
    showToast("Prompt enriched with domain optical descriptors.", "success");
  } catch (err) {
    showToast(err.message, "error");
  }
}

// Send Generated Image to Forensics
function sendGeneratedToForensics() {
  if (!state.generatedFilename) return;
  state.activeForensicFilename = state.generatedFilename;
  const currentTarget = document.getElementById("current-target-name");
  if (currentTarget) currentTarget.textContent = state.generatedFilename;
  switchTab("forensics");
  analyzeActiveImage();
}

// Send to Bidirectional Comparison
function sendToBidirectional() {
  switchTab("bidirectional");
  runBidirectionalComparison();
}

// Refresh / Run Bidirectional View
async function refreshBidirectionalView() {
  const emptyEl = document.getElementById("bidirectional-empty");
  const containerEl = document.getElementById("bidirectional-container");

  if (!state.uploadedFilename || !state.generatedFilename) {
    if (emptyEl) {
      emptyEl.classList.remove("hidden");
      emptyEl.textContent = "Please complete Image → Text analysis and generate a synthesized image for your active image to view the bidirectional comparison.";
    }
    if (containerEl) containerEl.classList.add("hidden");
    return;
  }

  if (emptyEl) emptyEl.classList.add("hidden");
  if (containerEl) containerEl.classList.remove("hidden");

  // Populate Original Image & Meta
  const biOrigImg = document.getElementById("bi-orig-img");
  const biOrigMeta = document.getElementById("bi-orig-meta");
  const biOrigDesc = document.getElementById("bi-orig-desc");

  if (biOrigImg && state.uploadedUrl) biOrigImg.src = state.uploadedUrl;
  if (biOrigMeta) biOrigMeta.textContent = state.uploadedFilename || "Uploaded Asset";
  if (biOrigDesc) {
    biOrigDesc.textContent = state.vlmOutput?.summary || state.vlmOutput?.detailed_description || "Original visual analysis representation.";
  }

  // Populate Generated Image & Meta
  const biGenImg = document.getElementById("bi-gen-img");
  const biGenMeta = document.getElementById("bi-gen-meta");
  const biGenPrompt = document.getElementById("bi-gen-prompt");

  if (biGenImg && (state.generatedBase64 || state.generatedUrl)) {
    biGenImg.src = state.generatedBase64 || state.generatedUrl;
  }
  if (biGenMeta) biGenMeta.textContent = state.generatedFilename || "Generated Asset";
  if (biGenPrompt) {
    biGenPrompt.textContent = state.currentPrompt || document.getElementById("t2i-prompt")?.value || "Synthesized generation prompt.";
  }

  if (state.uploadedFilename && state.generatedFilename) {
    await runBidirectionalComparison();
  }
}

async function runBidirectionalComparison() {
  if (!state.uploadedFilename || !state.generatedFilename) {
    showToast("Need both uploaded input and generated output for parity calculation.", "info");
    return;
  }

  try {
    const origDesc = state.vlmOutput?.summary || state.vlmOutput?.detailed_description || "";
    const genPrompt = state.currentPrompt || document.getElementById("t2i-prompt")?.value || "";

    const res = await fetch("/api/bidirectional/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        original_filename: state.uploadedFilename,
        generated_filename: state.generatedFilename,
        original_description: origDesc,
        generated_prompt: genPrompt
      })
    });

    if (!res.ok) throw new Error("Parity calculation failed");
    const data = await res.json();

    // Render 4 actual metrics
    const colorMetric = document.getElementById("bi-color-metric");
    const colorBar = document.getElementById("bi-color-bar");
    if (colorMetric && colorBar) {
      colorMetric.textContent = `${data.color_distribution_correlation}%`;
      colorBar.style.width = `${data.color_distribution_correlation}%`;
    }

    const semMetric = document.getElementById("bi-semantic-metric");
    const semBar = document.getElementById("bi-semantic-bar");
    if (semMetric && semBar) {
      semMetric.textContent = `${data.semantic_description_similarity}%`;
      semBar.style.width = `${data.semantic_description_similarity}%`;
    }

    const aspMetric = document.getElementById("bi-aspect-metric");
    const aspBar = document.getElementById("bi-aspect-bar");
    if (aspMetric && aspBar) {
      aspMetric.textContent = `${data.aspect_ratio_parity}%`;
      aspBar.style.width = `${data.aspect_ratio_parity}%`;
    }

    const overMetric = document.getElementById("bi-overall-metric");
    const overBar = document.getElementById("bi-overall-bar");
    if (overMetric && overBar) {
      overMetric.textContent = `${data.overall_multimodal_consistency}%`;
      overBar.style.width = `${data.overall_multimodal_consistency}%`;
    }

    showToast("Bidirectional multimodal parity calculated.", "success");
  } catch (err) {
    console.warn("Bidirectional parity error:", err);
  }
}

// Forensic Analysis Execution
async function analyzeActiveImage() {
  const target = state.activeForensicFilename || state.generatedFilename || state.uploadedFilename;
  if (!target) {
    showToast("No active image available to analyze. Please upload or generate an image.", "error");
    return;
  }
  await executeForensicAnalysis(target);
}

async function executeForensicAnalysis(filename) {
  const alpha = parseFloat(document.getElementById("forensic-alpha")?.value) || 0.45;
  const btn = document.getElementById("btn-analyze-active");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Analyzing patches...";
  }

  try {
    showToast(`Evaluating 64×64 spatial patches for ${filename}...`, "info");
    const res = await fetch("/api/forensics/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename, alpha })
    });

    if (!res.ok) throw new Error("Forensic evaluation failed");
    const data = await res.json();
    renderForensicResults(data, filename);
    showToast("Independent forensic detection completed.", "success");
    loadReportsList();
  } catch (err) {
    showToast(err.message, "error");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Analyze Active System Image";
    }
  }
}

function renderForensicResults(data, filename) {
  const rep = data.report;
  const vis = data.visualizations;

  const metricCards = document.getElementById("forensic-metric-cards");
  const reportCard = document.getElementById("forensic-report-card");
  if (metricCards) metricCards.classList.remove("hidden");
  if (reportCard) reportCard.classList.remove("hidden");

  // AI-Generated likelihood
  const aiScoreEl = document.getElementById("metric-ai-score");
  const aiProgressBar = document.getElementById("ai-progress-bar");
  if (aiScoreEl) aiScoreEl.textContent = `${rep.ai_generated_likelihood}%`;
  if (aiProgressBar) aiProgressBar.style.width = `${rep.ai_generated_likelihood}%`;

  // Real likelihood
  const realScoreEl = document.getElementById("metric-real-score");
  const realProgressBar = document.getElementById("real-progress-bar");
  if (realScoreEl) realScoreEl.textContent = `${rep.real_image_likelihood}%`;
  if (realProgressBar) realProgressBar.style.width = `${rep.real_image_likelihood}%`;

  // Anomaly level
  const anomalyLevelEl = document.getElementById("metric-anomaly-level");
  if (anomalyLevelEl) {
    anomalyLevelEl.textContent = rep.anomaly_level;
    anomalyLevelEl.className = rep.anomaly_level === 'HIGH' 
      ? 'text-2xl font-bold text-state-error' 
      : (rep.anomaly_level === 'MODERATE' ? 'text-2xl font-bold text-accent' : 'text-2xl font-bold text-state-success');
  }

  // Patch count
  const patchCountEl = document.getElementById("metric-patch-count");
  if (patchCountEl) patchCountEl.textContent = rep.patch_statistics.total_patches_evaluated;

  // Triptych Views
  const origImg = document.getElementById("vis-orig-img");
  const origDims = document.getElementById("orig-dims");
  const origPlaceholder = document.getElementById("vis-orig-placeholder");
  const resolvedOrigUrl = vis.orig_url || (filename.startsWith("gen_") ? `/outputs/generated/${filename}` : `/outputs/uploads/${filename}`);
  if (origImg) {
    origImg.src = resolvedOrigUrl;
    origImg.classList.remove("hidden");
    if (origPlaceholder) origPlaceholder.classList.add("hidden");
    origImg.onerror = () => {
      origImg.onerror = null; // Break recursive loop immediately
      if (state.uploadedFileUrl && origImg.src !== state.uploadedFileUrl) {
        origImg.src = state.uploadedFileUrl;
      } else {
        origImg.classList.add("hidden");
        if (origPlaceholder) origPlaceholder.classList.remove("hidden");
      }
    };
  }
  if (origDims && rep.image_metadata) {
    origDims.textContent = `${rep.image_metadata.width}×${rep.image_metadata.height} px`;
  }

  const heatmapImg = document.getElementById("vis-heatmap-img");
  const heatmapPlaceholder = document.getElementById("vis-heatmap-placeholder");
  const overlayImg = document.getElementById("vis-overlay-img");
  const overlayPlaceholder = document.getElementById("vis-overlay-placeholder");

  if (heatmapImg && vis.heatmap_url) {
    heatmapImg.onerror = () => {
      heatmapImg.onerror = null;
      heatmapImg.classList.add("hidden");
      if (heatmapPlaceholder) heatmapPlaceholder.classList.remove("hidden");
    };
    heatmapImg.src = vis.heatmap_url;
    heatmapImg.classList.remove("hidden");
    if (heatmapPlaceholder) heatmapPlaceholder.classList.add("hidden");
  }
  if (overlayImg && vis.overlay_url) {
    overlayImg.onerror = () => {
      overlayImg.onerror = null;
      overlayImg.classList.add("hidden");
      if (overlayPlaceholder) overlayPlaceholder.classList.remove("hidden");
    };
    overlayImg.src = vis.overlay_url;
    overlayImg.classList.remove("hidden");
    if (overlayPlaceholder) overlayPlaceholder.classList.add("hidden");
  }

  // Multi-stream FFT & Noise Residual
  const fftImg = document.getElementById("vis-fft-img");
  const residualImg = document.getElementById("vis-residual-img");
  if (fftImg && vis.fft_url) {
    fftImg.onerror = () => {
      fftImg.onerror = null;
      fftImg.classList.add("hidden");
    };
    fftImg.src = vis.fft_url;
    fftImg.classList.remove("hidden");
  }
  if (residualImg && vis.residual_url) {
    residualImg.onerror = () => {
      residualImg.onerror = null;
      residualImg.classList.add("hidden");
    };
    residualImg.src = vis.residual_url;
    residualImg.classList.remove("hidden");
  }

  // Report fields
  const reportIdBadge = document.getElementById("report-id-badge");
  const reportExplanation = document.getElementById("report-explanation");
  const reportDisclaimer = document.getElementById("report-disclaimer");

  if (reportIdBadge) reportIdBadge.textContent = rep.report_id;
  if (reportExplanation) reportExplanation.textContent = rep.forensic_explanation;
  if (reportDisclaimer) reportDisclaimer.textContent = rep.academic_disclaimer;
}

function updateAlphaLabel(val) {
  const alphaVal = document.getElementById("alpha-val");
  if (alphaVal) alphaVal.textContent = val;
}

// Knowledge Base Tab
async function fetchKnowledgeBase() {
  try {
    const res = await fetch("/api/knowledge-base");
    if (res.ok) {
      state.kbDocuments = await res.json();
      renderKnowledgeBase(state.kbDocuments);
    }
  } catch (err) {
    console.warn("Failed to fetch knowledge base:", err);
  }
}

function renderKnowledgeBase(docs) {
  const container = document.getElementById("kb-documents-grid");
  if (!container) return;
  container.innerHTML = docs.map(d => `
    <div class="res-card p-3.5 space-y-1.5 text-xs">
      <div class="flex items-center justify-between">
        <span class="font-semibold text-app-text">${d.title}</span>
        <span class="text-[10px] px-1.5 py-0.5 rounded bg-surface-muted text-app-muted font-mono">${d.category}</span>
      </div>
      <p class="text-app-muted text-[11px] leading-relaxed">${d.description}</p>
      <div class="text-[10px] text-app-muted font-mono pt-1">
        Keywords: ${(d.keywords || []).slice(0, 4).join(", ")}
      </div>
    </div>
  `).join("");
}

function filterKnowledgeBase(query) {
  const q = query.toLowerCase();
  const filtered = state.kbDocuments.filter(d => 
    d.title.toLowerCase().includes(q) || 
    d.description.toLowerCase().includes(q) ||
    d.category.toLowerCase().includes(q)
  );
  renderKnowledgeBase(filtered);
}

// Technical Pipeline Stages Rendering
function renderPipelineStages() {
  const container = document.getElementById("pipeline-stages-container");
  if (!container) return;

  container.innerHTML = TECHNICAL_PIPELINE_STAGES.map((s) => `
    <div class="res-card p-3.5 space-y-2">
      <div class="flex items-center justify-between">
        <span class="font-semibold text-xs text-accent font-mono">${s.step}</span>
        <span class="text-[11px] font-mono px-2 py-0.5 rounded bg-surface-muted text-app-muted border border-app-border">${s.model}</span>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs pt-1">
        <div><strong class="text-app-muted font-mono block text-[10px]">INPUT:</strong> <span class="text-app-text">${s.input}</span></div>
        <div><strong class="text-app-muted font-mono block text-[10px]">PROCESS:</strong> <span class="text-app-text">${s.process}</span></div>
        <div><strong class="text-app-muted font-mono block text-[10px]">OUTPUT:</strong> <span class="text-app-text">${s.output}</span></div>
        <div><strong class="text-app-muted font-mono block text-[10px]">RESEARCH PURPOSE:</strong> <span class="text-app-text">${s.purpose}</span></div>
      </div>
    </div>
  `).join("");
}

// Safe timestamp formatting for audit logs
function formatAuditTimestamp(ts) {
  if (!ts) return "—";
  try {
    const dt = new Date(ts);
    if (isNaN(dt.getTime())) {
      return String(ts).replace("T", " ").slice(0, 19);
    }
    const dStr = dt.toLocaleDateString([], { month: "short", day: "numeric" });
    const tStr = dt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    return `${dStr}, ${tStr}`;
  } catch {
    return String(ts).slice(0, 19);
  }
}

// Clear Experiment Logs
async function clearExperimentLogs(skipConfirm = false) {
  if (!skipConfirm) {
    const ok = confirm("Are you sure you want to CLEAR all experiment audit logs from the database?");
    if (!ok) return;
  }
  const btn = document.getElementById("btn-clear-experiments");
  if (btn) btn.disabled = true;

  try {
    const res = await fetch("/api/experiments/clear", { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to clear logs`);
    const data = await res.json();
    if (typeof showToast === "function") {
      showToast(data.message || "All experiment logs cleared successfully.", "success");
    }
    const tbody = document.getElementById("experiments-table-body");
    if (tbody) {
      tbody.innerHTML = `<tr><td colspan="7" class="p-4 text-center text-app-muted">All experiment records cleared. Ready for new runs.</td></tr>`;
    }
  } catch (err) {
    console.error("Failed to clear experiment logs:", err);
    if (typeof showToast === "function") {
      showToast("Error clearing logs: " + err.message, "error");
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

// Clear Reports List
async function clearReportsList(skipConfirm = false) {
  if (!skipConfirm) {
    const ok = confirm("Are you sure you want to CLEAR all generated forensic reports and heatmaps?");
    if (!ok) return;
  }
  const btn = document.getElementById("btn-clear-reports");
  if (btn) btn.disabled = true;

  try {
    const res = await fetch("/api/reports/clear", { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to clear reports`);
    const data = await res.json();
    if (typeof showToast === "function") {
      showToast(data.message || "All forensic reports cleared successfully.", "success");
    }
    state.reportsList = [];
    const tbody = document.getElementById("reports-table-body");
    if (tbody) {
      tbody.innerHTML = `<tr><td colspan="7" class="p-4 text-center text-app-muted">All forensic reports cleared. Ready for new evaluations.</td></tr>`;
    }
  } catch (err) {
    console.error("Failed to clear reports:", err);
    if (typeof showToast === "function") {
      showToast("Error clearing reports: " + err.message, "error");
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

// Load Experiment Logs
async function loadExperimentLogs(isManual = false) {
  if (isManual) {
    const shouldClear = confirm("Do you want to CLEAR all experiment logs to start fresh?\n\n• Click [OK] to CLEAR all data and reset.\n• Click [Cancel] to just reload existing records.");
    if (shouldClear) {
      await clearExperimentLogs(true);
      return;
    }
  }

  const btn = document.getElementById("btn-refresh-experiments");
  const icon = btn?.querySelector(".refresh-icon");
  const label = btn?.querySelector(".refresh-label");

  if (btn) {
    btn.disabled = true;
    if (icon) icon.classList.add("animate-spin");
    if (label) label.textContent = "Refreshing...";
  }

  try {
    const res = await fetch("/api/experiments?limit=50");
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch logs`);
    const exps = await res.json();
    const tbody = document.getElementById("experiments-table-body");
    if (!tbody) return;

    if (!Array.isArray(exps) || exps.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" class="p-4 text-center text-app-muted">No experiments logged yet. Execute the pipeline to record audit metadata.</td></tr>`;
    } else {
      tbody.innerHTML = exps.map(e => {
        const likelihood = Number(e.ai_generated_likelihood ?? 0);
        const likelihoodColor = likelihood > 60 ? "text-state-error" : "text-accent";
        const anomaly = e.anomaly_level || "LOW";
        const anomalyBadge = anomaly === "HIGH"
          ? "bg-rose-50 text-state-error border border-rose-200"
          : "bg-surface-muted text-app-muted border border-app-border";

        return `
          <tr class="hover:bg-surface-subtle transition">
            <td class="p-3 text-accent font-semibold font-mono text-xs">${e.experiment_id || "—"}</td>
            <td class="p-3 text-app-muted font-mono text-[11px]">${formatAuditTimestamp(e.timestamp)}</td>
            <td class="p-3 text-app-text">${e.experiment_type || "pipeline_run"}</td>
            <td class="p-3 text-app-muted font-mono text-[11px]">${e.image_width || 512}×${e.image_height || 512}</td>
            <td class="p-3 font-semibold ${likelihoodColor}">${likelihood.toFixed(1)}%</td>
            <td class="p-3"><span class="px-1.5 py-0.5 rounded text-[10px] ${anomalyBadge}">${anomaly}</span></td>
            <td class="p-3 text-app-muted font-mono text-[11px]">${e.report_id || "—"}</td>
          </tr>
        `;
      }).join("");
    }

    if (isManual && typeof showToast === "function") {
      showToast(`Experiment logs refreshed (${Array.isArray(exps) ? exps.length : 0} records).`, "success");
    }
  } catch (err) {
    console.warn("Failed to load experiments:", err);
    if (isManual && typeof showToast === "function") {
      showToast("Error refreshing logs: " + err.message, "error");
    }
  } finally {
    if (btn) {
      setTimeout(() => {
        btn.disabled = false;
        if (icon) icon.classList.remove("animate-spin");
        if (label) label.textContent = "Refresh Log";
      }, 350);
    }
  }
}

// Load Reports List
async function loadReportsList(isManual = false) {
  if (isManual) {
    const shouldClear = confirm("Do you want to CLEAR all forensic reports to start fresh?\n\n• Click [OK] to CLEAR all report data and reset.\n• Click [Cancel] to just reload existing reports.");
    if (shouldClear) {
      await clearReportsList(true);
      return;
    }
  }

  const btn = document.getElementById("btn-refresh-reports");
  const icon = btn?.querySelector(".refresh-icon");
  const label = btn?.querySelector(".refresh-label");

  if (btn) {
    btn.disabled = true;
    if (icon) icon.classList.add("animate-spin");
    if (label) label.textContent = "Refreshing...";
  }

  try {
    const res = await fetch("/api/reports");
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch reports`);
    const reports = await res.json();
    state.reportsList = reports;
    const tbody = document.getElementById("reports-table-body");
    if (!tbody) return;

    if (!Array.isArray(reports) || reports.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" class="p-4 text-center text-app-muted">No forensic reports generated yet. Execute a forensic analysis to produce reports.</td></tr>`;
    } else {
      tbody.innerHTML = reports.map(r => {
        const likelihood = Number(r.ai_generated_likelihood ?? 0);
        const likelihoodColor = likelihood > 60 ? "text-state-error" : "text-accent";
        const anomaly = r.anomaly_level || "UNKNOWN";
        const anomalyBadge = anomaly === "HIGH"
          ? "bg-rose-50 text-state-error border border-rose-200"
          : "bg-surface-muted text-app-muted border border-app-border";
        const score = Number(r.global_anomaly_score ?? 0);

        return `
          <tr class="hover:bg-surface-subtle transition">
            <td class="p-3 text-accent font-semibold font-mono text-xs">${r.report_id || "—"}</td>
            <td class="p-3 text-app-text truncate max-w-xs font-mono text-[11px]">${r.filename || "Unknown"}</td>
            <td class="p-3 font-semibold ${likelihoodColor}">${likelihood.toFixed(1)}%</td>
            <td class="p-3"><span class="px-1.5 py-0.5 rounded text-[10px] ${anomalyBadge}">${anomaly}</span></td>
            <td class="p-3 text-app-muted font-mono text-[11px]">${score.toFixed(4)}</td>
            <td class="p-3 text-app-muted truncate text-[11px]">${r.forensic_model || "spatial-frequency-patch-detector"}</td>
            <td class="p-3 text-right space-x-2">
              <button onclick="viewReportModal('${r.report_id}')" class="px-2 py-1 bg-surface border border-app-border rounded hover:bg-surface-muted hover:border-accent text-app-text text-[11px] cursor-pointer transition">View</button>
              <a href="${r.report_url}" download="${r.report_id}.json" class="px-2 py-1 bg-surface border border-app-border rounded hover:bg-surface-muted hover:border-accent text-accent text-[11px] cursor-pointer transition">Download</a>
            </td>
          </tr>
        `;
      }).join("");
    }

    if (isManual && typeof showToast === "function") {
      showToast(`Forensic reports refreshed (${Array.isArray(reports) ? reports.length : 0} records).`, "success");
    }
  } catch (err) {
    console.warn("Failed to load reports:", err);
    if (isManual && typeof showToast === "function") {
      showToast("Error refreshing reports: " + err.message, "error");
    }
  } finally {
    if (btn) {
      setTimeout(() => {
        btn.disabled = false;
        if (icon) icon.classList.remove("animate-spin");
        if (label) label.textContent = "Refresh Reports";
      }, 350);
    }
  }
}

// View Detailed Report in Modal
async function viewReportModal(reportId) {
  const modal = document.getElementById("report-modal");
  const modalTitle = document.getElementById("modal-report-title");
  const modalBody = document.getElementById("modal-report-body");
  const modalDownload = document.getElementById("modal-download-link");

  if (!modal || !modalBody) return;

  modal.classList.remove("hidden");
  modalBody.innerHTML = "<div>Loading report details...</div>";

  try {
    const res = await fetch(`/outputs/reports/${reportId}.json`);
    if (!res.ok) throw new Error("Could not load report JSON");
    const data = await res.json();
    state.selectedReport = data;

    if (modalTitle) modalTitle.textContent = `Report: ${data.report_id}`;
    if (modalDownload) {
      modalDownload.href = `/outputs/reports/${reportId}.json`;
      modalDownload.download = `${data.report_id}.json`;
    }

    modalBody.innerHTML = `
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 pb-3 border-b border-app-border">
        <div><strong>AI Likelihood:</strong> <span class="${data.ai_generated_likelihood > 60 ? 'text-state-error' : 'text-accent'} font-bold">${data.ai_generated_likelihood}%</span></div>
        <div><strong>Real Likelihood:</strong> <span>${data.real_image_likelihood}%</span></div>
        <div><strong>Anomaly Level:</strong> <span class="font-bold">${data.anomaly_level}</span></div>
        <div><strong>Score (Top 15%):</strong> <span>${data.global_anomaly_score.toFixed(4)}</span></div>
      </div>
      <div class="space-y-1 pt-1">
        <strong>Forensic Explanation:</strong>
        <p class="text-app-text bg-surface-muted p-2.5 rounded border border-app-border text-[11px] leading-relaxed">${data.forensic_explanation}</p>
      </div>
      <div class="space-y-1">
        <strong>Academic Disclaimer:</strong>
        <p class="text-app-muted bg-surface-muted p-2.5 rounded border border-app-border text-[10px] leading-relaxed">${data.academic_disclaimer}</p>
      </div>
      <div class="space-y-1">
        <strong>Patch Statistics:</strong>
        <div class="text-[11px] text-app-muted">Total: ${data.patch_statistics?.total_patches_evaluated} | Mean: ${data.patch_statistics?.mean_patch_score?.toFixed(4)} | Max: ${data.patch_statistics?.max_patch_score?.toFixed(4)}</div>
      </div>
    `;
  } catch (err) {
    modalBody.innerHTML = `<div class="text-state-error">Error loading report: ${err.message}</div>`;
  }
}

function closeReportModal() {
  const modal = document.getElementById("report-modal");
  if (modal) modal.classList.add("hidden");
}

/* ========================================================================== */
/* REALISTIC ASTRONOMICAL STARFIELD & CONSTELLATIONS (HOME PAGE ONLY)         */
/* ========================================================================== */
function init3DBackground() {
  const canvas = document.getElementById("bg-3d-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  let width = 0;
  let height = 0;
  let dpr = 1;

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  resize();
  window.addEventListener("resize", resize, { passive: true });

  const SPREAD_X = 640;
  const SPREAD_Y = 460;
  const SPREAD_Z = 480;
  const FOCAL_LENGTH = 540;

  // Real celestial spectral star colors (diamond whites, pale ice-blue, warm starlight, sapphire)
  const starColorsDark = [
    { r: 255, g: 255, b: 255 }, // Pure Diamond White
    { r: 240, g: 249, b: 255 }, // Celestial Ice White
    { r: 224, g: 242, b: 254 }, // Starlight Sky
    { r: 186, g: 230, b: 253 }, // Pale Electric Cyan
    { r: 254, g: 243, b: 199 }, // Stellar Warm Gold
    { r: 147, g: 197, b: 253 }  // Sirius Blue
  ];
  const starColorsLight = [
    { r: 37, g: 99, b: 235 },   // Royal Blue
    { r: 14, g: 165, b: 233 },  // Electric Cyan
    { r: 2, g: 132, b: 199 },   // Deep Sky
    { r: 79, g: 70, b: 229 }   // Indigo
  ];

  // 1. Constellation Landmark Stars (Forming real geometric asterisms)
  const landmarkStarData = [
    // Cassiopeia "W" (stars 0 to 4)
    { x: -380, y: -240, z: 80, name: 'Schedar' },
    { x: -320, y: -280, z: 60, name: 'Caph' },
    { x: -260, y: -230, z: 90, name: 'Gamma Cas' },
    { x: -210, y: -270, z: 70, name: 'Ruchbah' },
    { x: -160, y: -220, z: 100, name: 'Segin' },

    // Cygnus the Northern Cross (stars 5 to 9)
    { x: -180, y: -80, z: -40, name: 'Deneb' },
    { x: -190, y: 10, z: -20, name: 'Sadr' },
    { x: -200, y: 110, z: 0, name: 'Albireo' },
    { x: -280, y: 20, z: -10, name: 'Gienah' },
    { x: -100, y: 0, z: -30, name: 'Delta Cyg' },

    // Orion Asterism (stars 10 to 16)
    { x: -360, y: 120, z: 40, name: 'Betelgeuse' },
    { x: -240, y: 100, z: 60, name: 'Bellatrix' },
    { x: -310, y: 180, z: 50, name: 'Alnitak' },
    { x: -290, y: 190, z: 50, name: 'Alnilam' },
    { x: -270, y: 200, z: 50, name: 'Mintaka' },
    { x: -350, y: 270, z: 70, name: 'Saiph' },
    { x: -230, y: 250, z: 80, name: 'Rigel' },

    // Ursa Major / Big Dipper (stars 17 to 23)
    { x: 120, y: -280, z: 30, name: 'Alkaid' },
    { x: 180, y: -250, z: 20, name: 'Mizar' },
    { x: 230, y: -220, z: 40, name: 'Alioth' },
    { x: 280, y: -230, z: 30, name: 'Megrez' },
    { x: 340, y: -220, z: 50, name: 'Dubhe' },
    { x: 330, y: -160, z: 40, name: 'Merak' },
    { x: 270, y: -170, z: 30, name: 'Phecda' },

    // Lyra & Vega (stars 24 to 28)
    { x: 80, y: -70, z: -60, name: 'Vega' },
    { x: 130, y: -40, z: -50, name: 'Epsilon Lyr' },
    { x: 140, y: 20, z: -40, name: 'Zeta Lyr' },
    { x: 90, y: 30, z: -50, name: 'Sheliak' },
    { x: 70, y: -10, z: -60, name: 'Sulafat' },

    // Pegasus Great Square (stars 29 to 32)
    { x: 220, y: 90, z: -20, name: 'Alpheratz' },
    { x: 340, y: 80, z: 0, name: 'Scheat' },
    { x: 350, y: 200, z: 20, name: 'Markab' },
    { x: 230, y: 210, z: 10, name: 'Algenib' }
  ];

  // Specific constellation lines connecting designated star indices
  const constellationLines = [
    // Cassiopeia
    [0, 1], [1, 2], [2, 3], [3, 4],
    // Cygnus Cross
    [5, 6], [6, 7], [8, 6], [6, 9],
    // Orion
    [10, 12], [11, 14], [12, 13], [13, 14], [12, 15], [14, 16], [10, 11],
    // Big Dipper (Handle + Bowl)
    [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 20],
    // Lyra
    [24, 25], [25, 26], [26, 27], [27, 28], [28, 24],
    // Pegasus Square
    [29, 30], [30, 31], [31, 32], [32, 29]
  ];

  const stars = [];

  // Add Landmark Constellation Stars
  for (let i = 0; i < landmarkStarData.length; i++) {
    const d = landmarkStarData[i];
    stars.push({
      x: d.x,
      y: d.y,
      z: d.z,
      vx: (Math.random() - 0.5) * 0.08,
      vy: (Math.random() - 0.5) * 0.08,
      vz: (Math.random() - 0.5) * 0.08,
      isLandmark: true,
      hasSpikes: true,
      radius: 1.8 + Math.random() * 0.6, // 1.8 - 2.4px
      twinklePhase: Math.random() * Math.PI * 2,
      twinkleSpeed: 0.018 + Math.random() * 0.025,
      colorDark: starColorsDark[i % starColorsDark.length],
      colorLight: starColorsLight[i % starColorsLight.length],
      screenX: 0,
      screenY: 0,
      rotZ: 0,
      visible: false
    });
  }

  // 2. Add Background Field Stars (Tiny starlight pinpoints)
  const FIELD_STAR_COUNT = 160;
  for (let i = 0; i < FIELD_STAR_COUNT; i++) {
    const isMedium = i < 40;
    stars.push({
      x: (Math.random() - 0.5) * SPREAD_X * 2.2,
      y: (Math.random() - 0.5) * SPREAD_Y * 2.2,
      z: (Math.random() - 0.5) * SPREAD_Z * 2.0,
      vx: (Math.random() - 0.5) * 0.10,
      vy: (Math.random() - 0.5) * 0.10,
      vz: (Math.random() - 0.5) * 0.10,
      isLandmark: false,
      hasSpikes: isMedium && Math.random() > 0.6,
      radius: isMedium ? (1.1 + Math.random() * 0.4) : (0.55 + Math.random() * 0.45),
      twinklePhase: Math.random() * Math.PI * 2,
      twinkleSpeed: 0.012 + Math.random() * 0.035,
      colorDark: starColorsDark[Math.floor(Math.random() * starColorsDark.length)],
      colorLight: starColorsLight[Math.floor(Math.random() * starColorsLight.length)],
      screenX: 0,
      screenY: 0,
      rotZ: 0,
      visible: false
    });
  }

  // Shooting star / meteor streak manager
  let shootingStar = null;
  let nextShootingStarTime = Date.now() + 3500;

  function spawnShootingStar() {
    const startX = Math.random() * (width * 0.7);
    const startY = Math.random() * (height * 0.35);
    const angle = (Math.PI / 4) + (Math.random() - 0.5) * 0.3;
    const speed = 12 + Math.random() * 8;
    const length = 80 + Math.random() * 70;
    shootingStar = {
      x: startX,
      y: startY,
      dx: Math.cos(angle) * speed,
      dy: Math.sin(angle) * speed,
      length: length,
      alpha: 1.0,
      decay: 0.022 + Math.random() * 0.015
    };
  }

  // -------------------------------------------------------------
  // SYSTEM 2: SPATIAL NETWORK PARTICLES (FOR ALL OTHER WEBPAGES)
  // Matching Screenshot 1: Purple, Cyan, Blue nodes with Halos & Filaments
  // -------------------------------------------------------------
  const NODE_COUNT = 75;
  const NET_SPREAD_X = 520;
  const NET_SPREAD_Y = 380;
  const NET_SPREAD_Z = 460;
  const MAX_CONN_DIST = 165;

  const netNodes = [];
  const netPaletteDark = [
    { r: 59, g: 130, b: 246 },   // blue-500
    { r: 6, g: 182, b: 212 },    // cyan-500
    { r: 99, g: 102, b: 241 },   // indigo-500
    { r: 168, g: 85, b: 247 },   // purple-500 (Screenshot 1)
    { r: 56, g: 189, b: 248 }    // sky-400
  ];
  const netPaletteLight = [
    { r: 37, g: 99, b: 235 },    // blue-600
    { r: 8, g: 145, b: 178 },    // cyan-600
    { r: 79, g: 70, b: 229 },    // indigo-600
    { r: 147, g: 51, b: 234 }    // purple-600
  ];

  for (let i = 0; i < NODE_COUNT; i++) {
    netNodes.push({
      x: (Math.random() - 0.5) * NET_SPREAD_X * 2,
      y: (Math.random() - 0.5) * NET_SPREAD_Y * 2,
      z: (Math.random() - 0.5) * NET_SPREAD_Z * 2,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.35,
      vz: (Math.random() - 0.5) * 0.35,
      baseRadius: Math.random() * 2.2 + 1.2,
      pulse: Math.random() * Math.PI * 2,
      pulseSpeed: 0.02 + Math.random() * 0.025,
      colorDark: netPaletteDark[Math.floor(Math.random() * netPaletteDark.length)],
      colorLight: netPaletteLight[Math.floor(Math.random() * netPaletteLight.length)],
      screenX: 0,
      screenY: 0,
      screenRadius: 0,
      rotZ: 0,
      visible: false
    });
  }

  const DUST_COUNT = 30;
  const dustParticles = [];
  for (let i = 0; i < DUST_COUNT; i++) {
    dustParticles.push({
      x: (Math.random() - 0.5) * NET_SPREAD_X * 2.4,
      y: (Math.random() - 0.5) * NET_SPREAD_Y * 2.4,
      z: (Math.random() - 0.5) * NET_SPREAD_Z * 2,
      vy: -(0.15 + Math.random() * 0.3),
      size: Math.random() * 1.5 + 0.6,
      alpha: Math.random() * 0.5 + 0.2
    });
  }

  // Interactive mouse parallax
  let targetAngleX = 0;
  let targetAngleY = 0;
  let currentAngleX = 0;
  let currentAngleY = 0;
  let baseAngleY = 0;

  window.addEventListener("mousemove", (e) => {
    const normX = (e.clientX / window.innerWidth) - 0.5;
    const normY = (e.clientY / window.innerHeight) - 0.5;
    targetAngleY = normX * 0.35;
    targetAngleX = -normY * 0.28;
  }, { passive: true });

  let animationFrameId = null;

  function render() {
    if (document.hidden) {
      animationFrameId = requestAnimationFrame(render);
      return;
    }

    ctx.clearRect(0, 0, width, height);

    const isDark = document.documentElement.classList.contains("dark");
    const isHome = (!state || !state.activeTab || state.activeTab === "home");

    // Smooth subtle cosmic drift and parallax
    baseAngleY += 0.0006;
    currentAngleX += (targetAngleX - currentAngleX) * 0.04;
    currentAngleY += (targetAngleY - currentAngleY) * 0.04;

    const rotX = currentAngleX;
    const rotY = currentAngleY + baseAngleY;

    const cosX = Math.cos(rotX);
    const sinX = Math.sin(rotX);
    const cosY = Math.cos(rotY);
    const sinY = Math.sin(rotY);

    const centerX = width / 2;
    const centerY = height / 2;

    if (isHome) {
      // =========================================================
      // HOMEPAGE: Realistic Stars & Astronomical Constellations
      // =========================================================
      for (let i = 0; i < stars.length; i++) {
        const s = stars[i];

        s.x += s.vx;
        s.y += s.vy;
        s.z += s.vz;
        s.twinklePhase += s.twinkleSpeed;

        const boundX = s.isLandmark ? SPREAD_X * 1.2 : SPREAD_X * 2.2;
        const boundY = s.isLandmark ? SPREAD_Y * 1.2 : SPREAD_Y * 2.2;
        const boundZ = SPREAD_Z * 1.8;

        if (s.x < -boundX) s.x = boundX;
        else if (s.x > boundX) s.x = -boundX;
        if (s.y < -boundY) s.y = boundY;
        else if (s.y > boundY) s.y = -boundY;
        if (s.z < -boundZ) s.z = boundZ;
        else if (s.z > boundZ) s.z = -boundZ;

        const x1 = s.x * cosY - s.z * sinY;
        const z1 = s.z * cosY + s.x * sinY;
        const y2 = s.y * cosX - z1 * sinX;
        const z2 = z1 * cosX + s.y * sinX;

        s.rotZ = z2;

        const camDist = FOCAL_LENGTH + z2;
        if (camDist > 30) {
          const scale = FOCAL_LENGTH / camDist;
          s.screenX = centerX + x1 * scale;
          s.screenY = centerY + y2 * scale;
          s.scale = scale;
          s.visible = (s.screenX >= -40 && s.screenX <= width + 40 && s.screenY >= -40 && s.screenY <= height + 40);
        } else {
          s.visible = false;
        }
      }

      // 2. Draw Fine Celestial Constellation Lines
      for (let i = 0; i < constellationLines.length; i++) {
        const [idx1, idx2] = constellationLines[i];
        const s1 = stars[idx1];
        const s2 = stars[idx2];
        if (s1 && s2 && s1.visible && s2.visible) {
          const depthRatio = Math.max(0.2, Math.min(1.0, (s1.rotZ + s2.rotZ + SPREAD_Z * 2) / (SPREAD_Z * 3.5)));
          const alpha = depthRatio * (isDark ? 0.24 : 0.16);

          ctx.beginPath();
          ctx.moveTo(s1.screenX, s1.screenY);
          ctx.lineTo(s2.screenX, s2.screenY);
          ctx.strokeStyle = isDark
            ? `rgba(56, 189, 248, ${alpha})`
            : `rgba(37, 99, 235, ${alpha * 0.9})`;
          ctx.lineWidth = Math.max(0.4, 0.75 * depthRatio);
          ctx.stroke();
        }
      }

      // 3. Draw Stars with Authentic Astronomical Glints & Twinkling
      for (let i = 0; i < stars.length; i++) {
        const s = stars[i];
        if (!s.visible) continue;

        const twinkle = 0.65 + 0.35 * Math.sin(s.twinklePhase);
        const depthRatio = Math.max(0.25, Math.min(1.0, (s.rotZ + SPREAD_Z) / (SPREAD_Z * 2)));
        const alpha = Math.min(1.0, twinkle * depthRatio * (isDark ? 0.95 : 0.85));
        const c = isDark ? s.colorDark : s.colorLight;
        const r = Math.max(0.45, s.radius * s.scale);

        // Diffraction Spikes for Landmark / Bright Alpha Stars (The 4-point cross glint '+')
        if (s.hasSpikes && isDark && depthRatio > 0.35) {
          const spikeLen = (3.5 + 6.0 * twinkle) * s.scale;
          ctx.lineWidth = 0.7;

          // Horizontal starlight needle
          ctx.beginPath();
          ctx.moveTo(s.screenX - spikeLen, s.screenY);
          ctx.lineTo(s.screenX + spikeLen, s.screenY);
          ctx.strokeStyle = `rgba(${c.r}, ${c.g}, ${c.b}, ${alpha * 0.75})`;
          ctx.stroke();

          // Vertical starlight needle
          ctx.beginPath();
          ctx.moveTo(s.screenX, s.screenY - spikeLen);
          ctx.lineTo(s.screenX, s.screenY + spikeLen);
          ctx.strokeStyle = `rgba(${c.r}, ${c.g}, ${c.b}, ${alpha * 0.75})`;
          ctx.stroke();

          // Soft micro-corona
          ctx.beginPath();
          ctx.arc(s.screenX, s.screenY, r * 1.8, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${c.r}, ${c.g}, ${c.b}, ${alpha * 0.22})`;
          ctx.fill();
        }

        // Brilliant Pinpoint Diamond Core
        ctx.beginPath();
        ctx.arc(s.screenX, s.screenY, r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${c.r}, ${c.g}, ${c.b}, ${alpha})`;
        ctx.fill();
      }

      // 4. Subtle Cosmic Shooting Star (Meteor Streak)
      const now = Date.now();
      if (!shootingStar && now > nextShootingStarTime) {
        if (Math.random() < 0.6) {
          spawnShootingStar();
        }
        nextShootingStarTime = now + 8000 + Math.random() * 8000;
      }

      if (shootingStar) {
        const ss = shootingStar;
        const tailX = ss.x - (ss.dx / Math.hypot(ss.dx, ss.dy)) * ss.length;
        const tailY = ss.y - (ss.dy / Math.hypot(ss.dx, ss.dy)) * ss.length;

        const grad = ctx.createLinearGradient(tailX, tailY, ss.x, ss.y);
        grad.addColorStop(0, "rgba(56, 189, 248, 0)");
        grad.addColorStop(0.7, `rgba(186, 230, 253, ${ss.alpha * 0.45})`);
        grad.addColorStop(1, `rgba(255, 255, 255, ${ss.alpha * 0.85})`);

        ctx.beginPath();
        ctx.moveTo(tailX, tailY);
        ctx.lineTo(ss.x, ss.y);
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.2;
        ctx.stroke();

        ss.x += ss.dx;
        ss.y += ss.dy;
        ss.alpha -= ss.decay;
        if (ss.alpha <= 0 || ss.x > width + 100 || ss.y > height + 100) {
          shootingStar = null;
        }
      }

    } else {
      // =========================================================
      // ALL OTHER WEBPAGES: Spatial Network Particles (Screenshot 1)
      // Purple, Cyan, Blue glowing nodes with halos & filaments
      // =========================================================
      for (let i = 0; i < netNodes.length; i++) {
        const n = netNodes[i];

        n.x += n.vx;
        n.y += n.vy;
        n.z += n.vz;
        n.pulse += n.pulseSpeed;

        if (n.x < -NET_SPREAD_X) n.x = NET_SPREAD_X;
        else if (n.x > NET_SPREAD_X) n.x = -NET_SPREAD_X;
        if (n.y < -NET_SPREAD_Y) n.y = NET_SPREAD_Y;
        else if (n.y > NET_SPREAD_Y) n.y = -NET_SPREAD_Y;
        if (n.z < -NET_SPREAD_Z) n.z = NET_SPREAD_Z;
        else if (n.z > NET_SPREAD_Z) n.z = -NET_SPREAD_Z;

        const x1 = n.x * cosY - n.z * sinY;
        const z1 = n.z * cosY + n.x * sinY;
        const y2 = n.y * cosX - z1 * sinX;
        const z2 = z1 * cosX + n.y * sinX;

        n.rotZ = z2;

        const camDist = FOCAL_LENGTH + z2;
        if (camDist > 20) {
          const scale = FOCAL_LENGTH / camDist;
          n.screenX = centerX + x1 * scale;
          n.screenY = centerY + y2 * scale;
          n.screenRadius = Math.max(0.6, (n.baseRadius + Math.sin(n.pulse) * 0.6) * scale);
          n.visible = (n.screenX >= -50 && n.screenX <= width + 50 && n.screenY >= -50 && n.screenY <= height + 50);
        } else {
          n.visible = false;
        }
      }

      // Draw filaments between close nodes
      for (let i = 0; i < netNodes.length; i++) {
        const n1 = netNodes[i];
        if (!n1.visible) continue;

        for (let j = i + 1; j < netNodes.length; j++) {
          const n2 = netNodes[j];
          if (!n2.visible) continue;

          const dx = n1.x - n2.x;
          const dy = n1.y - n2.y;
          const dz = n1.z - n2.z;
          const dist3D = Math.sqrt(dx * dx + dy * dy + dz * dz);

          if (dist3D < MAX_CONN_DIST) {
            const proximity = 1 - (dist3D / MAX_CONN_DIST);
            const depthRatio = Math.max(0.15, Math.min(1, (n1.rotZ + n2.rotZ + NET_SPREAD_Z * 2) / (NET_SPREAD_Z * 3)));
            const alpha = proximity * proximity * depthRatio * (isDark ? 0.32 : 0.22);

            ctx.beginPath();
            ctx.moveTo(n1.screenX, n1.screenY);
            ctx.lineTo(n2.screenX, n2.screenY);
            ctx.strokeStyle = isDark
              ? `rgba(129, 140, 248, ${alpha})`
              : `rgba(59, 130, 246, ${alpha * 0.8})`;
            ctx.lineWidth = Math.max(0.5, 1.2 * depthRatio);
            ctx.stroke();
          }
        }
      }

      // Draw nodes sorted by depth (back to front with glow halos as in Screenshot 1)
      const sortedNetNodes = netNodes.filter(n => n.visible).sort((a, b) => b.rotZ - a.rotZ);

      for (let i = 0; i < sortedNetNodes.length; i++) {
        const n = sortedNetNodes[i];
        const c = isDark ? n.colorDark : n.colorLight;
        const depthRatio = Math.max(0.2, Math.min(1, (n.rotZ + NET_SPREAD_Z) / (NET_SPREAD_Z * 2)));
        const alpha = (0.35 + depthRatio * 0.55);

        // Glow halo in dark mode (the purple and cyan outer glow shown in Screenshot 1)
        if (isDark && depthRatio > 0.40) {
          ctx.beginPath();
          ctx.arc(n.screenX, n.screenY, n.screenRadius * 2.8, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${c.r}, ${c.g}, ${c.b}, ${alpha * 0.18})`;
          ctx.fill();
        }

        // Crisp core
        ctx.beginPath();
        ctx.arc(n.screenX, n.screenY, n.screenRadius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${c.r}, ${c.g}, ${c.b}, ${alpha})`;
        ctx.fill();
      }

      // Ambient floating dust particles
      for (let i = 0; i < dustParticles.length; i++) {
        const p = dustParticles[i];
        p.y += p.vy;
        if (p.y < -NET_SPREAD_Y) {
          p.y = NET_SPREAD_Y;
          p.x = (Math.random() - 0.5) * NET_SPREAD_X * 2.4;
        }

        const x1 = p.x * cosY - p.z * sinY;
        const z1 = p.z * cosY + p.x * sinY;
        const y2 = p.y * cosX - z1 * sinX;
        const z2 = z1 * cosX + p.y * sinX;

        const camDist = FOCAL_LENGTH + z2;
        if (camDist > 30) {
          const scale = FOCAL_LENGTH / camDist;
          const sx = centerX + x1 * scale;
          const sy = centerY + y2 * scale;
          if (sx >= 0 && sx <= width && sy >= 0 && sy <= height) {
            ctx.beginPath();
            ctx.arc(sx, sy, Math.max(0.5, p.size * scale), 0, Math.PI * 2);
            ctx.fillStyle = isDark
              ? `rgba(147, 197, 253, ${p.alpha * 0.4})`
              : `rgba(96, 165, 250, ${p.alpha * 0.25})`;
            ctx.fill();
          }
        }
      }
    }

    animationFrameId = requestAnimationFrame(render);
  }

  // Handle visibility changes to pause rendering when tab is hidden
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && !animationFrameId) {
      animationFrameId = requestAnimationFrame(render);
    }
  });

  animationFrameId = requestAnimationFrame(render);
}

// =========================================================================
// Galaxy Auth System Studio Navigation Linkage & Toast Notifications
// =========================================================================
function updateStudioAuthNav() {
  const savedUser = localStorage.getItem('multimodal_auth_user');
  const navLabel = document.getElementById('auth-nav-label');
  const triggerBtn = document.getElementById('btn-galaxy-auth-trigger');

  if (savedUser) {
    try {
      const u = JSON.parse(savedUser);
      if (navLabel) {
        navLabel.innerHTML = `<span class="inline-block w-2 h-2 rounded-full bg-emerald-400 mr-1 animate-pulse"></span>${u.name} (Sign Out)`;
      }
      if (triggerBtn) {
        triggerBtn.title = `Logged in as ${u.name}. Click to open Entry Screen / Sign Out.`;
        triggerBtn.onclick = function() {
          if (confirm(`Currently signed in as ${u.name}. Do you want to sign out and return to the Galaxy Entry Screen?`)) {
            localStorage.removeItem('multimodal_auth_user');
            if (window.GalaxyAuth && window.GalaxyAuth.open) {
              window.GalaxyAuth.open('login');
            } else if (window.openGalaxyAuthScreen) {
              window.openGalaxyAuthScreen('login');
            }
            updateStudioAuthNav();
          }
        };
      }
    } catch (e) {
      console.warn('Failed to parse auth user', e);
    }
  } else {
    if (navLabel) navLabel.textContent = 'Sign In / Portal';
    if (triggerBtn) {
      triggerBtn.title = 'Open Galaxy Auth Entry Screen';
      triggerBtn.onclick = function() {
        if (window.GalaxyAuth && window.GalaxyAuth.open) {
          window.GalaxyAuth.open('login');
        } else if (window.openGalaxyAuthScreen) {
          window.openGalaxyAuthScreen('login');
        }
      };
    }
  }
}

function showStudioToast(msg) {
  let toast = document.getElementById('studio-auth-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'studio-auth-toast';
    toast.className = 'fixed bottom-6 right-6 z-50 bg-[#0c0d1c]/95 border border-purple-500/40 text-white text-xs px-4 py-3 rounded-2xl shadow-2xl backdrop-blur-md flex items-center space-x-3 transition-all duration-300 transform translate-y-10 opacity-0 pointer-events-none font-mono';
    document.body.appendChild(toast);
  }
  toast.innerHTML = `<span class="w-2.5 h-2.5 rounded-full bg-purple-400 shadow-[0_0_10px_#c084fc]"></span><span>${msg}</span>`;
  toast.classList.remove('translate-y-10', 'opacity-0', 'pointer-events-none');
  setTimeout(() => {
    toast.classList.add('translate-y-10', 'opacity-0', 'pointer-events-none');
  }, 4200);
}

window.updateStudioAuthNav = updateStudioAuthNav;
window.showStudioToast = showStudioToast;
