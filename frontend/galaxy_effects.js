/**
 * Galaxy Theme Interactive Effects Engine
 * - Smooth page & workstation tab transitions with cosmic auroral sweeps
 * - Quirky, high-quality spring button animations and supernova click shockwaves
 */
(function () {
  'use strict';

  // =========================================================================
  // 1. UNIVERSAL SUPERNOVA CLICK SHOCKWAVE FOR ALL BUTTONS
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
  // 2. SMOOTH PAGE / TAB TRANSITION ENHANCER
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
      revealActiveTab();
    });
  } else {
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
