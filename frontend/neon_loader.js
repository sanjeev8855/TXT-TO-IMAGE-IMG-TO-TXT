/**
 * Neon Blue & Purple Parchment Scroll Image-to-Text Loading Screen Controller
 * Integrated from neon-image-to-text-loader-v8 for Image Analysis loading state
 *
 * Exact nib-to-line tracking: measures rendered text width & row baseline in real time
 * so the glowing nib stays pinned right on the active line and character it writes.
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
    { threshold: 25, label: 'RECOGNIZING OBJECTS & COMPOSITION...' },
    { threshold: 55, label: 'EXTRACTING COLOR & LIGHTING...' },
    { threshold: 80, label: 'TRANSCRIBING MULTIMODAL PROSE...' },
    { threshold: 95, label: 'FINALIZING ANALYSIS...' }
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
  let animationTimer = null;
  let particleAnimationId = null;
  let currentNibScreenPos = { x: window.innerWidth / 2, y: window.innerHeight / 2 };

  const NIB_OFFSET_X = (8 / 140) * 114;   // ~6.5px
  const NIB_OFFSET_Y = (232 / 240) * 196; // ~189.5px
  const NEON_COLORS = ['#00d4ff', '#c026d3', '#6366f1', '#e879f9', '#ffffff'];

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

  function runSequence() {
    if (!textLinesEl || !silhouetteSvg) return;
    textLinesEl.innerHTML = '';
    silhouetteSvg.classList.remove('dissolved');
    let lineIdx = 0;
    let charIdx = 0;

    const rowEls = SAMPLE_PAPER_TEXT.map((item) => {
      const div = document.createElement('div');
      div.className = `text-row ${item.cls}`;
      textLinesEl.appendChild(div);
      return div;
    });

    requestAnimationFrame(() => {
      if (rowEls[0]) positionQuillOnRow(rowEls[0], '', 0);
    });

    const totalChars = SAMPLE_PAPER_TEXT.reduce((acc, l) => acc + l.text.length, 0);
    let typedChars = 0;

    clearInterval(animationTimer);
    animationTimer = setInterval(() => {
      if (!isRunning) {
        clearInterval(animationTimer);
        return;
      }

      if (lineIdx >= SAMPLE_PAPER_TEXT.length) {
        const lastRow = rowEls[rowEls.length - 1];
        if (lastRow) positionQuillOnRow(lastRow, SAMPLE_PAPER_TEXT[SAMPLE_PAPER_TEXT.length - 1].text, 0);
        clearInterval(animationTimer);
        // Loop sequence gently if still running
        setTimeout(() => {
          if (isRunning) runSequence();
        }, 1600);
        return;
      }

      const currentLine = SAMPLE_PAPER_TEXT[lineIdx].text;
      const rowEl = rowEls[lineIdx];
      rowEl.classList.add('active');

      if (lineIdx >= 2) {
        silhouetteSvg.classList.add('dissolved');
      }

      charIdx++;
      typedChars++;
      const typedSlice = currentLine.slice(0, charIdx);
      rowEl.textContent = typedSlice;

      const wobble = Math.sin(typedChars * 0.95) * 1.6;
      positionQuillOnRow(rowEl, typedSlice, wobble);

      if (typedChars % 2 === 0) spawnQuillSparks(3);

      const pct = Math.min(100, Math.round((typedChars / totalChars) * 100));
      if (progressFill) progressFill.style.width = `${pct}%`;
      if (progressPercent) progressPercent.textContent = `${pct}%`;

      if (statusText) {
        for (let i = STATUS_PHASES.length - 1; i >= 0; i--) {
          if (pct >= STATUS_PHASES[i].threshold) {
            statusText.textContent = STATUS_PHASES[i].label;
            break;
          }
        }
      }

      if (charIdx >= currentLine.length) {
        lineIdx++;
        charIdx = 0;
      }
    }, 48);
  }

  // --- Public API for Image-to-Text Pipeline Integration ---
  window.NeonOCRLoader = {
    show(customMessage) {
      if (!loaderContainer) initElements();
      if (!loaderContainer) return;

      isRunning = true;
      loaderContainer.classList.remove('hidden');
      if (statusText && customMessage) {
        statusText.textContent = customMessage;
      }
      if (progressFill) progressFill.style.width = '0%';
      if (progressPercent) progressPercent.textContent = '0%';

      initParticles();
      cancelAnimationFrame(particleAnimationId);
      animateParticles();
      runSequence();
    },

    hide() {
      isRunning = false;
      clearInterval(animationTimer);
      cancelAnimationFrame(particleAnimationId);
      if (loaderContainer) {
        loaderContainer.classList.add('hidden');
      }
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initElements);
  } else {
    initElements();
  }
})();
