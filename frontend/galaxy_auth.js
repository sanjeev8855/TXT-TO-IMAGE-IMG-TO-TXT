/**
 * Galaxy Theme Animated Login & Sign Up Controller
 * Features fluid color-interpolated starfield particles, dual-layer sky & button crossfades,
 * spring height morphing, and an animated rocket launching success state.
 */
(function () {
  const body = document.body;
  const authCard = document.getElementById('authCard');
  const formsViewport = document.getElementById('formsViewport');
  const loginForm = document.getElementById('loginForm');
  const signupForm = document.getElementById('signupForm');
  const loginError = document.getElementById('loginError');
  const signupError = document.getElementById('signupError');
  const loginBtn = document.getElementById('loginBtn');
  const signupBtn = document.getElementById('signupBtn');
  const formBody = document.getElementById('formBody');
  const successOverlay = document.getElementById('successOverlay');
  const successTitle = document.getElementById('successTitle');
  const successSub = document.getElementById('successSub');
  const resetDemoBtn = document.getElementById('resetDemoBtn');
  const switchPrompt = document.getElementById('switchPrompt');
  const switchModeBtn = document.getElementById('switchModeBtn');
  const socialBtns = document.querySelectorAll('.social-btn');

  let currentMode = 'login'; // 'login' (purple) | 'signup' (orange)

  // --- 1. Fluid Color-Interpolated Starfield Canvas ---
  const canvas = document.getElementById('galaxyCanvas');
  const ctx = canvas.getContext('2d');
  let stars = [];
  let themeProgress = 0; // 0 = purple, 1 = orange

  function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  for (let i = 0; i < 75; i++) {
    stars.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      r: Math.random() * 1.6 + 0.4,
      vy: -Math.random() * 0.22 - 0.05,
      alpha: Math.random() * 0.7 + 0.2,
      pulse: Math.random() * Math.PI * 2
    });
  }

  function lerp(a, b, t) {
    return Math.round(a + (b - a) * t);
  }

  function renderGalaxy() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const targetProgress = currentMode === 'signup' ? 1 : 0;
    themeProgress += (targetProgress - themeProgress) * 0.055;

    const fillR = lerp(233, 254, themeProgress);
    const fillG = lerp(213, 215, themeProgress);
    const fillB = lerp(255, 170, themeProgress);

    const glowR = lerp(168, 249, themeProgress);
    const glowG = lerp(85, 115, themeProgress);
    const glowB = lerp(247, 22, themeProgress);

    const starFill = `rgb(${fillR}, ${fillG}, ${fillB})`;
    const starGlow = `rgb(${glowR}, ${glowG}, ${glowB})`;

    for (const s of stars) {
      s.y += s.vy;
      s.pulse += 0.025;
      if (s.y < -5) {
        s.y = canvas.height + 5;
        s.x = Math.random() * canvas.width;
      }
      const opacity = s.alpha * (0.65 + 0.35 * Math.sin(s.pulse));
      ctx.save();
      ctx.globalAlpha = opacity;
      ctx.fillStyle = starFill;
      ctx.shadowBlur = 7;
      ctx.shadowColor = starGlow;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
    requestAnimationFrame(renderGalaxy);
  }
  renderGalaxy();

  // --- 2. Smooth Form Viewport Height Sync ---
  function syncViewportHeight() {
    const activeForm = currentMode === 'login' ? loginForm : signupForm;
    formsViewport.style.height = `${activeForm.offsetHeight}px`;
  }
  window.addEventListener('resize', syncViewportHeight);
  requestAnimationFrame(syncViewportHeight);

  // --- 3. Silk-Smooth Switch Between Purple Login & Orange Sign Up ---
  function setAuthMode(mode) {
    if (mode === currentMode) return;
    currentMode = mode;
    const isSignup = mode === 'signup';

    authCard.classList.add('morph-pulse');
    setTimeout(() => authCard.classList.remove('morph-pulse'), 240);

    body.setAttribute('data-theme', isSignup ? 'orange' : 'purple');

    if (isSignup) {
      loginForm.classList.remove('active');
      signupForm.classList.add('active');
      switchPrompt.textContent = 'Already have an account?';
      switchModeBtn.textContent = 'Login';
      switchModeBtn.setAttribute('href', '#login');
    } else {
      signupForm.classList.remove('active');
      loginForm.classList.add('active');
      switchPrompt.textContent = "Don't have an account?";
      switchModeBtn.textContent = 'Sign up';
      switchModeBtn.setAttribute('href', '#signup');
    }

    clearErrors();
    requestAnimationFrame(syncViewportHeight);
  }

  switchModeBtn.addEventListener('click', (e) => {
    e.preventDefault();
    setAuthMode(currentMode === 'login' ? 'signup' : 'login');
  });

  // --- 4. Password Visibility Toggle ---
  document.querySelectorAll('.toggle-pass-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const targetInput = document.getElementById(btn.dataset.target);
      if (!targetInput) return;
      const isPassword = targetInput.type === 'password';
      targetInput.type = isPassword ? 'text' : 'password';
      btn.setAttribute('aria-label', isPassword ? 'Hide password' : 'Show password');
      btn.innerHTML = isPassword
        ? '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path><line x1="1" y1="1" x2="23" y2="23"></line></svg>'
        : '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>';
    });
  });

  // --- 5. Validation & Rocket Launch Success Overlay ---
  function triggerError(bannerEl, message) {
    bannerEl.textContent = message;
    bannerEl.classList.add('visible');
    syncViewportHeight();
    authCard.classList.remove('shake');
    void authCard.offsetWidth;
    authCard.classList.add('shake');
  }

  function clearErrors() {
    [loginError, signupError].forEach((el) => {
      el.textContent = '';
      el.classList.remove('visible');
    });
    syncViewportHeight();
  }

  document.querySelectorAll('.auth-form input').forEach((inp) => {
    inp.addEventListener('input', clearErrors);
  });

  function showSuccessState(titleText, subtitleText, userName = null) {
    successTitle.textContent = titleText;
    successSub.textContent = subtitleText;
    formBody.classList.add('fade-out');
    successOverlay.classList.add('active');
    successOverlay.setAttribute('aria-hidden', 'false');

    window.dispatchEvent(
      new CustomEvent('galaxyAuthSuccess', {
        detail: { mode: currentMode, name: userName || 'Dr. Sanjeev' }
      })
    );
  }

  function resetAuthState() {
    loginBtn.classList.remove('loading');
    signupBtn.classList.remove('loading');
    formBody.classList.remove('fade-out');
    successOverlay.classList.remove('active');
    successOverlay.setAttribute('aria-hidden', 'true');
    clearErrors();
  }

  loginForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const userVal = document.getElementById('loginUsername').value.trim();
    const passVal = document.getElementById('loginPassword').value.trim();

    if (!userVal) {
      triggerError(loginError, 'Please enter your username or email.');
      document.getElementById('loginUsername').focus();
      return;
    }
    if (!passVal) {
      triggerError(loginError, 'Please enter your password.');
      document.getElementById('loginPassword').focus();
      return;
    }

    clearErrors();
    loginBtn.classList.add('loading');
    setTimeout(() => {
      loginBtn.classList.remove('loading');
      showSuccessState('Liftoff Successful!', 'Launching into orbit...', userVal || 'Dr. Sanjeev');
    }, 850);
  });

  signupForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const nameVal = document.getElementById('signupName').value.trim();
    const emailVal = document.getElementById('signupEmail').value.trim();
    const passVal = document.getElementById('signupPassword').value.trim();

    if (!nameVal) {
      triggerError(signupError, 'Please enter your full name.');
      document.getElementById('signupName').focus();
      return;
    }
    if (!emailVal || !emailVal.includes('@')) {
      triggerError(signupError, 'Please enter a valid email address.');
      document.getElementById('signupEmail').focus();
      return;
    }
    if (passVal.length < 6) {
      triggerError(signupError, 'Password must be at least 6 characters.');
      document.getElementById('signupPassword').focus();
      return;
    }

    clearErrors();
    signupBtn.classList.add('loading');
    setTimeout(() => {
      signupBtn.classList.remove('loading');
      showSuccessState('Liftoff Successful!', 'Account launched into orbit!', nameVal || 'Dr. Sanjeev');
    }, 850);
  });

  socialBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const provider = btn.dataset.provider || 'Account';
      const activeBtn = currentMode === 'login' ? loginBtn : signupBtn;
      activeBtn.classList.add('loading');
      setTimeout(() => {
        activeBtn.classList.remove('loading');
        showSuccessState(
          'Liftoff Successful!',
          `Connected via ${provider}. Launching into orbit...`,
          `Dr. Sanjeev (${provider})`
        );
      }, 700);
    });
  });

  resetDemoBtn.addEventListener('click', resetAuthState);

  window.demoLoginSanjeev = function () {
    const userInp = document.getElementById('loginUsername');
    const passInp = document.getElementById('loginPassword');
    if (userInp) userInp.value = 'sanjeev@multimodal-ai.org';
    if (passInp) passInp.value = 'studio2026demo';
    clearErrors();
    loginBtn.classList.add('loading');
    setTimeout(() => {
      loginBtn.classList.remove('loading');
      showSuccessState('Liftoff Successful!', 'Welcome Dr. Sanjeev! Launching into orbit...', 'Dr. Sanjeev');
    }, 650);
  };

  window.enterStudioAsGuest = function () {
    const entryScreen = document.getElementById('galaxy-auth-entry-screen');
    if (entryScreen) {
      entryScreen.classList.add('dismissed');
      localStorage.setItem('multimodal_auth_user', JSON.stringify({
        authenticated: true,
        mode: 'guest',
        name: 'Guest Scholar',
        role: 'Research Scholar'
      }));
      if (window.updateStudioAuthNav) window.updateStudioAuthNav();
      if (window.showStudioToast) window.showStudioToast('⚡ Entered Multimodal Studio as Guest.');
    }
  };

  window.openGalaxyAuthScreen = function (mode = 'login') {
    const entryScreen = document.getElementById('galaxy-auth-entry-screen');
    if (entryScreen) {
      entryScreen.classList.remove('dismissed');
      resetAuthState();
      setAuthMode(mode);
    }
  };

  window.GalaxyAuth = {
    setMode: setAuthMode,
    showSuccess: showSuccessState,
    reset: resetAuthState,
    demoLogin: window.demoLoginSanjeev,
    guestPass: window.enterStudioAsGuest,
    open: window.openGalaxyAuthScreen
  };

  // Rocket Liftoff Complete -> Transition Entry Screen into Studio
  window.addEventListener('galaxyAuthSuccess', (e) => {
    setTimeout(() => {
      const entryScreen = document.getElementById('galaxy-auth-entry-screen');
      if (entryScreen) {
        entryScreen.classList.add('dismissed');
      }
      const rawName = e.detail && e.detail.name ? e.detail.name : 'Dr. Sanjeev';
      const cleanName = rawName.includes('@') ? (rawName.split('@')[0].charAt(0).toUpperCase() + rawName.split('@')[0].slice(1)) : rawName;
      localStorage.setItem('multimodal_auth_user', JSON.stringify({
        authenticated: true,
        mode: e.detail.mode,
        name: cleanName,
        role: 'Lead AI Architect'
      }));
      if (window.updateStudioAuthNav) window.updateStudioAuthNav();
      if (window.showStudioToast) window.showStudioToast(`🚀 Liftoff complete! Welcome to Multimodal Studio, ${cleanName}.`);
    }, 1500);
  });
})();
