/**
 * Galaxy Theme Interactive Effects Engine
 * - Smooth page & workstation tab transitions with cosmic auroral sweeps
 * - Quirky, high-quality spring button animations and supernova click shockwaves
 * - Celestial Galaxy Cursor with gravitational aura and stardust particle trails
 */
(function () {
  'use strict';

  // =========================================================================
  // 1. CELESTIAL GALAXY CURSOR WITH STARDUST TRAILS & HOVER STATES
  // =========================================================================
  const isFinePointer = window.matchMedia('(hover: hover) and (pointer: fine)').matches;

  let dotEl = null;
  let auraEl = null;
  let trailCanvas = null;
  let trailCtx = null;
  let stardustParticles = [];

  let mouseX = -100;
  let mouseY = -100;
  let auraX = -100;
  let auraY = -100;
  let isMoving = false;
  let moveTimeout = null;
  let lastParticleDist = 0;
  let prevMouseX = -100;
  let prevMouseY = -100;

  function initGalaxyCursor() {
    if (!isFinePointer) return;

    // Center Core Dot
    dotEl = document.createElement('div');
    dotEl.className = 'galaxy-cursor-dot';
    document.body.appendChild(dotEl);

    // Orbiting Nebula Aura Ring
    auraEl = document.createElement('div');
    auraEl.className = 'galaxy-cursor-aura';
    document.body.appendChild(auraEl);

    // Stardust Particle Canvas
    trailCanvas = document.createElement('canvas');
    trailCanvas.className = 'galaxy-cursor-trail-canvas';
    document.body.appendChild(trailCanvas);
    trailCtx = trailCanvas.getContext('2d');

    function resizeTrailCanvas() {
      trailCanvas.width = window.innerWidth;
      trailCanvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resizeTrailCanvas, { passive: true });
    resizeTrailCanvas();

    document.body.classList.add('galaxy-cursor-active');

    // Mouse Tracking
    window.addEventListener('mousemove', (e) => {
      mouseX = e.clientX;
      mouseY = e.clientY;

      if (!dotEl.classList.contains('visible')) {
        dotEl.classList.add('visible');
        auraEl.classList.add('visible');
        auraX = mouseX;
        auraY = mouseY;
      }

      // Check distance moved to emit stardust
      const dist = Math.hypot(mouseX - prevMouseX, mouseY - prevMouseY);
      if (dist > 14 && stardustParticles.length < 24) {
        spawnStardust(mouseX, mouseY);
        prevMouseX = mouseX;
        prevMouseY = mouseY;
      }

      clearTimeout(moveTimeout);
      moveTimeout = setTimeout(() => {
        isMoving = false;
      }, 150);
      isMoving = true;
    }, { passive: true });

    // Window Entry / Exit
    document.addEventListener('mouseleave', () => {
      if (dotEl) dotEl.classList.remove('visible');
      if (auraEl) auraEl.classList.remove('visible');
    });

    document.addEventListener('mouseenter', () => {
      if (dotEl) dotEl.classList.add('visible');
      if (auraEl) auraEl.classList.add('visible');
    });

    // Mousedown / Mouseup Click Snap
    window.addEventListener('mousedown', () => {
      if (auraEl) auraEl.classList.add('clicking');
      if (dotEl) dotEl.classList.add('clicking');
    }, { passive: true });

    window.addEventListener('mouseup', () => {
      if (auraEl) auraEl.classList.remove('clicking');
      if (dotEl) dotEl.classList.remove('clicking');
    }, { passive: true });

    // Quirky Hover Detection via Event Delegation
    document.addEventListener('mouseover', (e) => {
      if (!auraEl || !dotEl) return;
      const target = e.target;

      // 1. Globe Canvas / Scene
      if (target.closest('#globe-3d-canvas, #orb, .orb-scene')) {
        auraEl.classList.add('hover-globe');
        dotEl.classList.add('hover-globe');
        auraEl.classList.remove('hover-button', 'hover-card');
        dotEl.classList.remove('hover-button', 'hover-card');
        return;
      }

      // 2. Interactive Buttons / Nav Tabs / Action Pills
      if (target.closest('button, .btn-acid, .nav-tab, .btn-clear-pill, .social-btn, .button, .navcta, .primary-btn, [role="button"]')) {
        auraEl.classList.add('hover-button');
        dotEl.classList.add('hover-button');
        auraEl.classList.remove('hover-globe', 'hover-card');
        dotEl.classList.remove('hover-globe', 'hover-card');
        return;
      }

      // 3. Bento Cards / Links / Inputs
      if (target.closest('.res-card, a, input, textarea, select, .cursor-pointer')) {
        auraEl.classList.add('hover-card');
        dotEl.classList.add('hover-card');
        auraEl.classList.remove('hover-button', 'hover-globe');
        dotEl.classList.remove('hover-button', 'hover-globe');
        return;
      }

      // Default state
      auraEl.classList.remove('hover-button', 'hover-globe', 'hover-card');
      dotEl.classList.remove('hover-button', 'hover-globe', 'hover-card');
    }, { passive: true });

    requestAnimationFrame(renderCursorPhysics);
  }

  // Stardust Particle Emitter
  const COSMIC_COLORS = [
    'rgba(56, 189, 248, ',   // Electric cyan
    'rgba(168, 85, 247, ',   // Cosmic violet
    'rgba(249, 115, 22, ',   // Stellar orange
    'rgba(255, 255, 255, ',  // Pure white star
    'rgba(99, 102, 241, '    // Deep indigo
  ];

  function spawnStardust(x, y) {
    const angle = Math.random() * Math.PI * 2;
    const speed = Math.random() * 0.9 + 0.2;
    const baseColor = COSMIC_COLORS[Math.floor(Math.random() * COSMIC_COLORS.length)];
    stardustParticles.push({
      x: x + (Math.random() - 0.5) * 8,
      y: y + (Math.random() - 0.5) * 8,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed - 0.15,
      radius: Math.random() * 1.8 + 0.7,
      alpha: 0.85,
      decay: Math.random() * 0.035 + 0.025,
      color: baseColor
    });
  }

  function renderCursorPhysics() {
    if (dotEl && auraEl) {
      // Instant Core Dot
      dotEl.style.left = mouseX + 'px';
      dotEl.style.top = mouseY + 'px';

      // Silky Spring / Lerp for Orbiting Nebula Aura Ring
      auraX += (mouseX - auraX) * 0.19;
      auraY += (mouseY - auraY) * 0.19;
      auraEl.style.left = auraX + 'px';
      auraEl.style.top = auraY + 'px';
    }

    // Render Stardust Canvas
    if (trailCtx && trailCanvas) {
      trailCtx.clearRect(0, 0, trailCanvas.width, trailCanvas.height);
      for (let i = stardustParticles.length - 1; i >= 0; i--) {
        const p = stardustParticles[i];
        p.x += p.vx;
        p.y += p.vy;
        p.alpha -= p.decay;

        if (p.alpha <= 0) {
          stardustParticles.splice(i, 1);
          continue;
        }

        trailCtx.save();
        trailCtx.beginPath();
        trailCtx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        trailCtx.fillStyle = p.color + p.alpha + ')';
        trailCtx.shadowBlur = 6;
        trailCtx.shadowColor = p.color + '0.8)';
        trailCtx.fill();
        trailCtx.restore();
      }
    }

    requestAnimationFrame(renderCursorPhysics);
  }


  // =========================================================================
  // 2. UNIVERSAL SUPERNOVA CLICK SHOCKWAVE FOR ALL BUTTONS
  // =========================================================================
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('button, .btn-acid, .button, .navcta, .nav-tab, .btn-clear-pill, .social-btn, .primary-btn, a[role="button"]');
    if (!btn) return;

    const rect = btn.getBoundingClientRect();
    const ripple = document.createElement('span');
    ripple.className = 'galaxy-cosmic-ripple';

    const diameter = Math.max(rect.width, rect.height) * 1.6;
    ripple.style.width = diameter + 'px';
    ripple.style.height = diameter + 'px';

    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    ripple.style.left = clickX + 'px';
    ripple.style.top = clickY + 'px';

    btn.appendChild(ripple);

    setTimeout(() => {
      ripple.remove();
    }, 600);
  }, { passive: true });


  // =========================================================================
  // 3. SMOOTH PAGE / TAB TRANSITION ENHANCER
  // =========================================================================
  function revealActiveTab() {
    const activeTabId = window.state ? window.state.activeTab : 'home';
    const activeEl = document.getElementById(`tab-${activeTabId || 'home'}`);
    if (activeEl) {
      activeEl.classList.add('tab-revealed');
    }

    const heroSection = document.getElementById('hero-section');
    if (heroSection && (!activeTabId || activeTabId === 'home')) {
      heroSection.classList.add('hero-revealed');
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      initGalaxyCursor();
      revealActiveTab();
    });
  } else {
    initGalaxyCursor();
    revealActiveTab();
  }

  // Hook global transition helper
  window.revealGalaxyTab = function (tabElement) {
    if (!tabElement) return;
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        tabElement.classList.add('tab-revealed');
      });
    });
  };

})();
