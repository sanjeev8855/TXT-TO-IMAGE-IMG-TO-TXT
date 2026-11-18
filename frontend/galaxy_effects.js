/**
 * Galaxy Theme Interactive Effects Engine
 * - Smooth page & workstation tab transitions with cosmic auroral sweeps
 * - Quirky, high-quality spring button animations and supernova click shockwaves
 * - Celestial Shooting Star Hover Engine (Streaks across buttons and night sky on hover)
 */
(function () {
  'use strict';

  // =========================================================================
  // 1. CELESTIAL SHOOTING STAR HOVER ENGINE
  // =========================================================================
  let starCanvas = null;
  let starCtx = null;
  let activeMeteors = [];
  let starSparks = [];
  let lastHoverTime = 0;

  function initShootingStarEngine() {
    // Background Shooting Star Canvas
    starCanvas = document.createElement('canvas');
    starCanvas.className = 'galaxy-shooting-star-canvas';
    document.body.appendChild(starCanvas);
    starCtx = starCanvas.getContext('2d');

    function resizeStarCanvas() {
      starCanvas.width = window.innerWidth;
      starCanvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resizeStarCanvas, { passive: true });
    resizeStarCanvas();

    // Trigger Shooting Star on Hover over any button, tab, card, link, or dropzone
    document.addEventListener('mouseover', (e) => {
      const target = e.target.closest('button, .btn-acid, .nav-tab, .btn-clear-pill, .social-btn, .button, .navcta, .primary-btn, a, .res-card, #drop-zone, .cursor-pointer, [role="button"]');
      if (!target) return;

      const now = performance.now();
      // Throttle slightly so rapid movement generates clean, elegant shooting stars
      if (now - lastHoverTime < 180) return;
      lastHoverTime = now;

      const rect = target.getBoundingClientRect();
      const originX = rect.left + rect.width * 0.5 + (Math.random() - 0.5) * (rect.width * 0.4);
      const originY = rect.top + rect.height * 0.5 + (Math.random() - 0.5) * (rect.height * 0.4);

      shootCosmicStar(originX, originY);
    }, { passive: true });

    // Also occasionally shoot a faint background shooting star during mouse movements across dark space
    let lastMoveTime = 0;
    window.addEventListener('mousemove', (e) => {
      const now = performance.now();
      if (now - lastMoveTime > 3200 && Math.random() < 0.35) {
        lastMoveTime = now;
        shootCosmicStar(e.clientX, e.clientY, true);
      }
    }, { passive: true });

    requestAnimationFrame(renderShootingStars);
  }

  // Meteor / Shooting Star Object Factory
  function shootCosmicStar(targetX, targetY, isAmbient = false) {
    if (activeMeteors.length >= 8) return;

    // Angle: downward diagonal streak (-35 deg to -42 deg)
    const angleRad = (-36 + (Math.random() - 0.5) * 8) * (Math.PI / 180);
    const speed = isAmbient ? (Math.random() * 8 + 14) : (Math.random() * 10 + 16);
    const length = isAmbient ? (Math.random() * 70 + 80) : (Math.random() * 90 + 110);

    // Start slightly upstream along the flight vector
    const startOffset = isAmbient ? 160 : 70;
    const startX = targetX - Math.cos(angleRad) * startOffset;
    const startY = targetY - Math.sin(angleRad) * startOffset;

    activeMeteors.push({
      x: startX,
      y: startY,
      vx: Math.cos(angleRad) * speed,
      vy: Math.sin(angleRad) * speed,
      length: length,
      alpha: 1.0,
      fadeRate: isAmbient ? 0.032 : 0.042,
      thickness: isAmbient ? 1.4 : 2.2,
      sparkTimer: 0
    });
  }

  // Render Loop for Shooting Stars & Stardust Sparks
  function renderShootingStars() {
    if (starCtx && starCanvas) {
      starCtx.clearRect(0, 0, starCanvas.width, starCanvas.height);

      // 1. Update and Render Active Shooting Stars
      for (let i = activeMeteors.length - 1; i >= 0; i--) {
        const m = activeMeteors[i];
        m.x += m.vx;
        m.y += m.vy;
        m.alpha -= m.fadeRate;

        if (m.alpha <= 0 || m.x > starCanvas.width + 100 || m.y > starCanvas.height + 100) {
          activeMeteors.splice(i, 1);
          continue;
        }

        // Emit micro stardust sparks along the shooting star's path
        m.sparkTimer++;
        if (m.sparkTimer % 2 === 0 && starSparks.length < 35) {
          starSparks.push({
            x: m.x + (Math.random() - 0.5) * 4,
            y: m.y + (Math.random() - 0.5) * 4,
            vx: (Math.random() - 0.5) * 0.8 - m.vx * 0.08,
            vy: (Math.random() - 0.5) * 0.8 - m.vy * 0.08,
            alpha: m.alpha * 0.8,
            decay: Math.random() * 0.04 + 0.03,
            radius: Math.random() * 1.5 + 0.8
          });
        }

        // Draw Shooting Star Tail
        const tailX = m.x - (m.vx / Math.hypot(m.vx, m.vy)) * m.length;
        const tailY = m.y - (m.vy / Math.hypot(m.vx, m.vy)) * m.length;

        const grad = starCtx.createLinearGradient(tailX, tailY, m.x, m.y);
        grad.addColorStop(0, 'rgba(168, 85, 247, 0)');
        grad.addColorStop(0.35, `rgba(168, 85, 247, ${m.alpha * 0.35})`);
        grad.addColorStop(0.75, `rgba(56, 189, 248, ${m.alpha * 0.85})`);
        grad.addColorStop(1, `rgba(255, 255, 255, ${m.alpha})`);

        starCtx.save();
        starCtx.beginPath();
        starCtx.moveTo(tailX, tailY);
        starCtx.lineTo(m.x, m.y);
        starCtx.strokeStyle = grad;
        starCtx.lineWidth = m.thickness;
        starCtx.lineCap = 'round';
        starCtx.shadowBlur = 10;
        starCtx.shadowColor = '#38bdf8';
        starCtx.stroke();

        // Draw Radiant Meteor Head
        starCtx.beginPath();
        starCtx.arc(m.x, m.y, m.thickness * 1.2, 0, Math.PI * 2);
        starCtx.fillStyle = `rgba(255, 255, 255, ${m.alpha})`;
        starCtx.shadowBlur = 14;
        starCtx.shadowColor = '#ffffff';
        starCtx.fill();
        starCtx.restore();
      }

      // 2. Update and Render Trailing Stardust Sparks
      for (let j = starSparks.length - 1; j >= 0; j--) {
        const sp = starSparks[j];
        sp.x += sp.vx;
        sp.y += sp.vy;
        sp.alpha -= sp.decay;

        if (sp.alpha <= 0) {
          starSparks.splice(j, 1);
          continue;
        }

        starCtx.save();
        starCtx.beginPath();
        starCtx.arc(sp.x, sp.y, sp.radius, 0, Math.PI * 2);
        starCtx.fillStyle = `rgba(56, 189, 248, ${sp.alpha})`;
        starCtx.shadowBlur = 5;
        starCtx.shadowColor = 'rgba(168, 85, 247, 0.8)';
        starCtx.fill();
        starCtx.restore();
      }
    }

    requestAnimationFrame(renderShootingStars);
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

    // Also trigger a shooting star right from the click location!
    shootCosmicStar(e.clientX, e.clientY);

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
      initShootingStarEngine();
      revealActiveTab();
    });
  } else {
    initShootingStarEngine();
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
