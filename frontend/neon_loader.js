/**
 * Neon Blue & Purple Parchment Scroll Image-to-Text Loading Screen Controller
 * Integrated from neon-image-to-text-loader-v8 for Image Analysis loading state
 *
 * Synchronized Progress Engine:
 * - Accurately reflects analysis progression from 1% to 100% matching real backend execution
 * - Exact nib-to-line tracking: measures rendered text width & row baseline in real time
 * - Never resets or loops mid-analysis; smoothly reaches 100% upon completion
 */
(function () {
  'use strict';

  const SAMPLE_PAPER_TEXT = [
    { text: 'Visual Analysis Active', cls: 'title' },
    { text: 'Parsing pixel distributions,', cls: '' },
    { text: 'extracting semantic scene context,', cls: '' },
    { text: 'identifying objects & lighting,', cls: '' },
    { text: 'synthesizing multimodal caption...', cls: 'purple-ink' },
    { text: 'Finalizing vision output.', cls: 'title' }
  ];

  const STATUS_PHASES = [
    { threshold: 0, label: 'SCANNING IMAGE DETAILS...' },
    { threshold: 20, label: 'ANALYZING IMAGE & EXTRACTING VISUAL FEATURES...' },
    { threshold: 58, label: 'QUERYING VECTOR RAG KNOWLEDGE BASE...' },
    { threshold: 75, label: 'SYNTHESIZING PROMPT & COMPOSITION BOUNDARIES...' },
    { threshold: 96, label: 'FINALIZING VISION TRANSCRIPTION (100%)...' }
  ];

  let loaderContainer = null;
  let stageEl = null;
  let textLinesEl = null;
  let silhouetteSvg = null;
  let neonQuill = null;
  let statusText = null;
  let progressFill = null;
  let progressPercent = null;
  let canvas = null;
  let ctx = null;

  // Offscreen canvas to measure exact rendered pixel width of typed text
  const measureCanvas = document.createElement('canvas');
  const measureCtx = measureCanvas.getContext('2d');

  let particles = [];
  let isRunning = false;
  let renderLoopId = null;
  let particleAnimationId = null;
  let currentNibScreenPos = { x: window.innerWidth / 2, y: window.innerHeight / 2 };

  const NIB_OFFSET_X = (8 / 140) * 114;   // ~6.5px
  const NIB_OFFSET_Y = (232 / 240) * 196; // ~189.5px
  const NEON_COLORS = ['#00d4ff', '#c026d3', '#6366f1', '#e879f9', '#ffffff'];

  // Total character count across all manuscript lines
  const TOTAL_CHARS = SAMPLE_PAPER_TEXT.reduce((acc, l) => acc + l.text.length, 0);

  // Dynamic Progress Engine State
  let currentPercent = 1;
  let targetPercent = 5;
  let stageCeiling = 18;
  let creepSpeedPerSec = 4.5;
  let lastFrameTime = performance.now();
  let renderedChars = 0;
  let rowEls = [];

  function initElements() {
    loaderContainer = document.getElementById('neonLoaderContainer');
    stageEl = document.getElementById('neonStage');
    textLinesEl = document.getElementById('neonTextLines');
    silhouetteSvg = document.getElementById('neonSilhouetteSvg');
    neonQuill = document.getElementById('neonQuill');
    statusText = document.getElementById('neonStatusText');
    progressFill = document.getElementById('neonProgressFill');
    progressPercent = document.getElementById('neonProgressPercent');
    canvas = document.getElementById('neonParticleCanvas');

    if (canvas) {
      ctx = canvas.getContext('2d');
      resizeCanvas();
      window.addEventListener('resize', resizeCanvas, { passive: true });
    }
  }

  function resizeCanvas() {
    if (!canvas) return;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  function initParticles() {
    if (!canvas) return;
    particles = [];
    for (let i = 0; i < 65; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        r: Math.random() * 2.2 + 0.6,
        vx: (Math.random() - 0.5) * 0.45,
        vy: -Math.random() * 0.65 - 0.15,
        color: NEON_COLORS[Math.floor(Math.random() * NEON_COLORS.length)],
        alpha: Math.random() * 0.8 + 0.2
      });
    }
  }

  function spawnQuillSparks(count = 4) {
    if (!canvas) return;
    for (let i = 0; i < count; i++) {
      particles.push({
        x: currentNibScreenPos.x + (Math.random() - 0.5) * 8,
        y: currentNibScreenPos.y + (Math.random() - 0.5) * 8,
        r: Math.random() * 2.4 + 0.8,
        vx: (Math.random() - 0.5) * 2.2,
        vy: -Math.random() * 2.4 - 0.5,
        color: Math.random() > 0.45 ? '#e879f9' : '#00d4ff',
        alpha: 1,
        decay: 0.025
      });
    }
  }

  function animateParticles() {
    if (!isRunning || !ctx || !canvas) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;
      if (p.decay) {
        p.alpha -= p.decay;
        if (p.alpha <= 0) {
          particles.splice(i, 1);
          continue;
        }
      } else {
        if (p.y < -10) p.y = canvas.height + 10;
        if (p.x < -10) p.x = canvas.width + 10;
        if (p.x > canvas.width + 10) p.x = -10;
      }

      ctx.save();
      ctx.globalAlpha = Math.max(0, p.alpha);
      ctx.fillStyle = p.color;
      ctx.shadowBlur = 10;
      ctx.shadowColor = p.color;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
    particleAnimationId = requestAnimationFrame(animateParticles);
  }

  function positionQuillOnRow(rowEl, typedSlice, wobble = 0) {
    if (!stageEl || !rowEl || !neonQuill) return;
    const stageRect = stageEl.getBoundingClientRect();
    const rowRect = rowEl.getBoundingClientRect();
    if (!stageRect.width || !rowRect.width) return;

    const scaleX = stageRect.width / stageEl.offsetWidth || 1;
    const scaleY = stageRect.height / stageEl.offsetHeight || 1;

    const computed = window.getComputedStyle(rowEl);
    measureCtx.font = `${computed.fontWeight} ${computed.fontSize} ${computed.fontFamily}`;
    const textWidth = measureCtx.measureText(typedSlice).width;

    const tiltRad = (-5 * Math.PI) / 180;
    const clampedX = Math.min(textWidth, rowEl.clientWidth - 2);
    const localX = clampedX * Math.cos(tiltRad);
    const localY = clampedX * Math.sin(tiltRad) + rowEl.offsetHeight * 0.74;

    const rowLeftInStage = (rowRect.left - stageRect.left) / scaleX;
    const rowTopInStage = (rowRect.top - stageRect.top) / scaleY;

    const targetNibX = rowLeftInStage + localX;
    const targetNibY = rowTopInStage + localY + wobble;

    currentNibScreenPos.x = stageRect.left + targetNibX * scaleX;
    currentNibScreenPos.y = stageRect.top + targetNibY * scaleY;

    const quillX = targetNibX - NIB_OFFSET_X;
    const quillY = targetNibY - NIB_OFFSET_Y;
    const angleDeg = wobble * 0.85;

    neonQuill.style.transform = `translate(${quillX.toFixed(1)}px, ${quillY.toFixed(1)}px) rotate(${angleDeg.toFixed(2)}deg)`;
  }

  /**
   * Updates parchment manuscript text rows and quill position to match current character count
   */
  function syncManuscriptToChars(targetChars) {
    if (!rowEls || !rowEls.length) return;

    let accumulated = 0;
    let activeRowEl = rowEls[0];
    let activeSlice = '';
    let activeLineIdx = 0;

    for (let i = 0; i < SAMPLE_PAPER_TEXT.length; i++) {
      const lineText = SAMPLE_PAPER_TEXT[i].text;
      const rowEl = rowEls[i];
      const prevTotal = accumulated;
      accumulated += lineText.length;

      if (targetChars <= prevTotal) {
        rowEl.textContent = '';
        rowEl.classList.remove('active');
      } else if (targetChars >= accumulated) {
        rowEl.textContent = lineText;
        rowEl.classList.remove('active');
        activeRowEl = rowEl;
        activeSlice = lineText;
        activeLineIdx = i;
      } else {
        const sliceLen = targetChars - prevTotal;
        const slice = lineText.slice(0, sliceLen);
        rowEl.textContent = slice;
        rowEl.classList.add('active');
        activeRowEl = rowEl;
        activeSlice = slice;
        activeLineIdx = i;
      }
    }

    if (silhouetteSvg) {
      if (activeLineIdx >= 2) {
        silhouetteSvg.classList.add('dissolved');
      } else {
        silhouetteSvg.classList.remove('dissolved');
      }
    }

    const wobble = Math.sin(targetChars * 0.95) * 1.5;
    positionQuillOnRow(activeRowEl, activeSlice, wobble);

    if (targetChars > renderedChars) {
      spawnQuillSparks(2);
    }
    renderedChars = targetChars;
  }

  /**
   * Main 60fps Animation Loop:
   * Smoothly eases currentPercent toward targetPercent while pacing steadily,
   * keeping the quill, manuscript lines, and 1% - 100% progress bar in exact sync.
   */
  function animationStep(now) {
    if (!isRunning) return;

    const dt = Math.min(0.1, (now - lastFrameTime) / 1000);
    lastFrameTime = now;

    // Smooth asymptotic creep towards stageCeiling while waiting for backend response
    if (targetPercent < stageCeiling) {
      targetPercent = Math.min(stageCeiling, targetPercent + creepSpeedPerSec * dt);
    }

    // Smooth lerp of current percentage towards target percentage
    const diff = targetPercent - currentPercent;
    if (Math.abs(diff) > 0.05) {
      const lerpFactor = diff > 0 ? Math.min(1, dt * 7.5) : Math.min(1, dt * 10);
      currentPercent += diff * lerpFactor;
    } else {
      currentPercent = targetPercent;
    }

    const clampedPercent = Math.max(1, Math.min(100, currentPercent));
    const intPercent = Math.round(clampedPercent);

    // Update Progress UI
    if (progressFill) {
      progressFill.style.width = `${clampedPercent.toFixed(1)}%`;
    }
    if (progressPercent) {
      progressPercent.textContent = `${intPercent}%`;
    }

    // Update Status Phase Label
    if (statusText && !statusText.dataset.manual) {
      for (let i = STATUS_PHASES.length - 1; i >= 0; i--) {
        if (intPercent >= STATUS_PHASES[i].threshold) {
          statusText.textContent = STATUS_PHASES[i].label;
          break;
        }
      }
    }

    // Sync manuscript text and cyber-quill nib to current progress
    const targetCharCount = Math.min(TOTAL_CHARS, Math.floor((clampedPercent / 100) * TOTAL_CHARS));
    syncManuscriptToChars(targetCharCount);

    renderLoopId = requestAnimationFrame(animationStep);
  }

  function setupManuscriptDOM() {
    if (!textLinesEl) return;
    textLinesEl.innerHTML = '';
    rowEls = SAMPLE_PAPER_TEXT.map((item) => {
      const div = document.createElement('div');
      div.className = `text-row ${item.cls}`;
      textLinesEl.appendChild(div);
      return div;
    });

    if (rowEls[0]) {
      positionQuillOnRow(rowEls[0], '', 0);
    }
  }

  // --- Public API for Image-to-Text Pipeline Integration ---
  window.NeonOCRLoader = {
    /**
     * Shows loader and starts synchronized progression from 1%
     */
    show(customMessage) {
      if (!loaderContainer) initElements();
      if (!loaderContainer) return;

      isRunning = true;
      currentPercent = 1;
      targetPercent = 6;
      stageCeiling = 16;
      creepSpeedPerSec = 4.0;
      renderedChars = 0;
      lastFrameTime = performance.now();

      loaderContainer.classList.remove('hidden');

      if (statusText) {
        statusText.dataset.manual = customMessage ? '1' : '';
        statusText.textContent = customMessage || STATUS_PHASES[0].label;
      }
      if (progressFill) progressFill.style.width = '1%';
      if (progressPercent) progressPercent.textContent = '1%';

      setupManuscriptDOM();
      initParticles();

      cancelAnimationFrame(particleAnimationId);
      cancelAnimationFrame(renderLoopId);

      animateParticles();
      renderLoopId = requestAnimationFrame(animationStep);
    },

    /**
     * Sets a milestone progress percentage and status message corresponding to real backend phases
     * @param {number} percent Target percentage (e.g. 15, 60, 78)
     * @param {string} [message] Status description
     * @param {number} [ceiling] Upper bound while waiting for the next response
     * @param {number} [speed] Creep rate in percent per second
     */
    setProgress(percent, message, ceiling = null, speed = null) {
      if (!isRunning) return;
      targetPercent = Math.max(targetPercent, percent);
      if (ceiling !== null) stageCeiling = ceiling;
      if (speed !== null) creepSpeedPerSec = speed;
      if (statusText && message) {
        statusText.dataset.manual = '1';
        statusText.textContent = message;
      }
    },

    /**
     * Smoothly finishes progress to 100%, displays final manuscript line and holds briefly
     * @param {string} [finalMessage] Optional completion message
     * @returns {Promise<void>}
     */
    complete(finalMessage = 'VISUAL ANALYSIS & SYNTHESIS COMPLETE (100%)') {
      if (!isRunning) return Promise.resolve();

      targetPercent = 100;
      stageCeiling = 100;
      creepSpeedPerSec = 40.0;

      if (statusText) {
        statusText.dataset.manual = '1';
        statusText.textContent = finalMessage;
      }

      return new Promise((resolve) => {
        const checkDone = () => {
          if (!isRunning) {
            resolve();
            return;
          }
          if (currentPercent >= 99.5) {
            currentPercent = 100;
            if (progressFill) progressFill.style.width = '100%';
            if (progressPercent) progressPercent.textContent = '100%';
            syncManuscriptToChars(TOTAL_CHARS);

            // Brief satisfying pause at 100% so user registers completion
            setTimeout(() => {
              resolve();
            }, 450);
          } else {
            setTimeout(checkDone, 50);
          }
        };
        checkDone();
      });
    },

    /**
     * Hides the loading screen
     */
    hide() {
      isRunning = false;
      cancelAnimationFrame(renderLoopId);
      cancelAnimationFrame(particleAnimationId);
      if (loaderContainer) {
        loaderContainer.classList.add('hidden');
      }
      if (statusText) {
        delete statusText.dataset.manual;
      }
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initElements);
  } else {
    initElements();
  }
})();
