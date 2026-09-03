/**
 * AI Retouch Studio Pro — Core Web Application Logic
 * MS Office Fluent Ribbon UI & Interactive Blemish Management Engine
 */

(function () {
  'use strict';

  // --- DEFAULT CONFIGURATION & STATE ---
  const DEFAULT_KEY = '';

  const state = {
    serverUrl: localStorage.getItem('ai_server_url') || 'http://127.0.0.1:8765',
    apiKey: localStorage.getItem('ai_gemini_key') || DEFAULT_KEY,
    theme: localStorage.getItem('ai_studio_theme') || 'dark',
    activeRibbonTab: 'home',

    originalImage: null,       // HTMLImageElement
    originalFile: null,        // File
    originalWidth: 0,
    originalHeight: 0,

    retouchedImage: null,      // HTMLImageElement
    skinMaskImage: null,       // HTMLImageElement
    skinMaskBlob: null,

    blobs: [],                 // Array of blemish detections
    spotHistory: [],           // Undo/Redo history stack
    spotHistoryIndex: -1,
    spotToolMode: 'toggle',    // 'toggle' | 'add' | 'erase'
    spotBrushRadius: 6,        // Default radius when adding spots (3 to 18)
    hoveredBlobIndex: -1,
    selectedBlobIndex: -1,     // Selected spot for inspection
    isDraggingErase: false,
    cursorImgPos: { x: 0, y: 0, onCanvas: false },

    baseToneRGB: [210, 170, 150],
    baseToneLAB: [170, 140, 140],
    fitzpatrickType: 'Type III',

    // Tools & Viewport
    activeTool: 'blemish',     // 'blemish' | 'eyedropper'
    viewMode: 'split',         // 'split' | 'after' | 'before'
    splitPercent: 50.0,        // 0 to 100
    isDraggingSplit: false,

    zoom: 1.0,
    panX: 0,
    panY: 0,
    isPanning: false,
    startPanX: 0,
    startPanY: 0,

    showSpots: true,
    showSkinMask: false,

    // Retouch Controls Parameters
    params: {
      includeHeal: true,
      sensitivity: 0.50,
      healMode: 'full_inpaint',
      preserveMoles: true,
      preserveFreckles: false,
      textureBlend: 0.25,
      featherRadius: 3,
      grainIntensity: 0.03,

      includeDb: true,
      dbStrength: 0.40,

      includeSmooth: true,
      smoothStrength: 0.45,
      textureKeep: 0.40,

      includeLighten: true,
      lightenStrength: 0.35,

      includeEyesTeeth: false,
      teethWhiten: 0.50,
      eyeBrighten: 0.45,

      includeShine: false,
      shineStrength: 0.40
    },

    isAnalyzing: false,
    isPreviewing: false,
    isExporting: false,
    previewDebounceTimer: null,
    previewAbortController: null
  };

  // --- DOM ELEMENTS ---
  const elements = {
    // Ribbon Tabs & Panels
    ribbonTabs: document.querySelectorAll('.ribbon-tab'),
    ribbonPanels: {
      home: document.getElementById('ribbon-panel-home'),
      blemish: document.getElementById('ribbon-panel-blemish'),
      skin: document.getElementById('ribbon-panel-skin'),
      features: document.getElementById('ribbon-panel-features'),
      view: document.getElementById('ribbon-panel-view')
    },

    // Dropzone & Viewport Stage
    uploadDropzone: document.getElementById('upload-dropzone'),
    fileInput: document.getElementById('file-input'),
    btnBrowseFile: document.getElementById('btn-browse-file'),
    btnBrowseFileCenter: document.getElementById('btn-browse-file-center'),
    canvasContainer: document.getElementById('canvas-container'),
    viewportStage: document.getElementById('viewport-stage'),

    canvasBefore: document.getElementById('canvas-before'),
    canvasAfter: document.getElementById('canvas-after'),
    canvasOverlay: document.getElementById('canvas-overlay'),
    splitDivider: document.getElementById('split-divider'),
    splitHandle: document.getElementById('split-handle'),

    processingBadge: document.getElementById('processing-badge'),
    processingBadgeText: document.getElementById('processing-badge-text'),
    spotTooltip: document.getElementById('spot-tooltip'),
    spotTooltipText: document.getElementById('spot-tooltip-text'),

    // Toast Container & Custom Modal Dialog
    toastContainer: document.getElementById('toast-container'),
    modalCustomDialog: document.getElementById('modal-custom-dialog'),
    dialogTitle: document.getElementById('dialog-title'),
    dialogMessage: document.getElementById('dialog-message'),
    dialogIconContainer: document.getElementById('dialog-icon-container'),
    dialogIcon: document.getElementById('dialog-icon'),
    btnDialogCancel: document.getElementById('btn-dialog-cancel'),
    btnDialogConfirm: document.getElementById('btn-dialog-confirm'),

    // Status Bar
    serverStatusDot: document.getElementById('server-status-dot'),
    serverStatusText: document.getElementById('server-status-text'),
    geminiStatusPill: document.getElementById('gemini-status-pill'),
    geminiStatusText: document.getElementById('gemini-status-text'),
    labelImageDim: document.getElementById('label-image-dim'),
    labelSpotsCount: document.getElementById('label-spots-count'),
    labelSkinToneContainer: document.getElementById('label-skin-tone-container'),
    toneSwatchBottom: document.getElementById('tone-swatch-bottom'),
    labelSkinToneText: document.getElementById('label-skin-tone-text'),
    labelRenderTime: document.getElementById('label-render-time'),
    labelZoomLevel: document.getElementById('label-zoom-level'),

    // Main Tools & View Switcher
    toolBtnBlemish: document.getElementById('tool-btn-blemish'),
    toolBtnEyedropper: document.getElementById('tool-btn-eyedropper'),
    viewModeSplit: document.getElementById('view-mode-split'),
    viewModeAfter: document.getElementById('view-mode-after'),
    viewModeBefore: document.getElementById('view-mode-before'),
    chkShowSpots: document.getElementById('chk-show-spots'),
    chkShowSkinMask: document.getElementById('chk-show-skin-mask'),
    btnToggleSpotsBlemish: document.getElementById('btn-toggle-spots-blemish'),
    btnToggleSkinBlemish: document.getElementById('btn-toggle-skin-blemish'),

    // Spot Sub-Modes & Radius Controls
    btnModeToggle: document.getElementById('btn-mode-toggle'),
    btnModeAdd: document.getElementById('btn-mode-add'),
    btnModeErase: document.getElementById('btn-mode-erase'),
    btnCleanOverlaps: document.getElementById('btn-clean-overlaps'),
    labelSpotRadiusToolbar: document.getElementById('label-spot-radius-toolbar'),
    btnSpotRadiusDec: document.getElementById('btn-spot-radius-dec'),
    btnSpotRadiusInc: document.getElementById('btn-spot-radius-inc'),
    sliderSpotRadius: document.getElementById('slider-spot-radius'),
    btnSpotsSelectAll: document.getElementById('btn-spots-select-all'),
    btnSpotsDeselectAll: document.getElementById('btn-spots-deselect-all'),
    btnSpotsClearAll: document.getElementById('btn-spots-clear-all'),
    btnUndoSpots: document.getElementById('btn-undo-spots'),
    btnRedoSpots: document.getElementById('btn-redo-spots'),

    // Selected Spot Inspector
    spotInspectorCard: document.getElementById('spot-inspector-card'),
    labelInspectorSpotId: document.getElementById('label-inspector-spot-id'),
    btnInspectorClose: document.getElementById('btn-inspector-close'),
    labelInspectorCoords: document.getElementById('label-inspector-coords'),
    sliderInspectorRadius: document.getElementById('slider-inspector-radius'),
    valInspectorRadius: document.getElementById('val-inspector-radius'),
    btnInspectorToggle: document.getElementById('btn-inspector-toggle'),
    btnInspectorDelete: document.getElementById('btn-inspector-delete'),

    // Zoom Controls
    btnZoomIn: document.getElementById('btn-zoom-in'),
    btnZoomOut: document.getElementById('btn-zoom-out'),
    btnZoomFit: document.getElementById('btn-zoom-fit'),

    // Hero & Refinement Actions
    btnAnalyze: document.getElementById('btn-analyze'),
    btnAnalyzeText: document.getElementById('btn-analyze-text'),
    btnAnalyzeBlemish: document.getElementById('btn-analyze-blemish'),
    btnAnalyzeBlemishText: document.getElementById('btn-analyze-blemish-text'),
    btnApplyHealing: document.getElementById('btn-apply-healing'),
    btnApplyHealingHome: document.getElementById('btn-apply-healing-home'),
    sliderSensitivity: document.getElementById('slider-sensitivity'),
    valSensitivity: document.getElementById('val-sensitivity'),
    inputPrompt: document.getElementById('input-prompt'),
    btnSendPrompt: document.getElementById('btn-send-prompt'),
    selectPreset: document.getElementById('select-preset'),
    btnResetImage: document.getElementById('btn-reset-image'),

    // Export Elements
    btnExportMain: document.getElementById('btn-export-main'),
    exportBtnLabel: document.getElementById('export-btn-label'),

    // Theme & Modals
    btnThemeToggle: document.getElementById('btn-theme-toggle'),
    selectThemePalette: document.getElementById('select-theme-palette'),
    btnShortcuts: document.getElementById('btn-shortcuts'),
    modalShortcuts: document.getElementById('modal-shortcuts'),
    btnCloseShortcuts: document.getElementById('btn-close-shortcuts'),
    modalSettings: document.getElementById('modal-settings'),
    btnSettings: document.getElementById('btn-settings'),
    btnCloseSettings: document.getElementById('btn-close-settings'),
    btnSaveSettings: document.getElementById('btn-save-settings'),
    inputApiKey: document.getElementById('input-api-key'),
    inputServerUrl: document.getElementById('input-server-url'),

    // Healing Parameters
    selectHealMode: document.getElementById('select-heal-mode'),
    chkPreserveMoles: document.getElementById('chk-preserve-moles'),
    chkPreserveFreckles: document.getElementById('chk-preserve-freckles'),
    sliderTextureBlend: document.getElementById('slider-texture-blend'),
    valTextureBlend: document.getElementById('val-texture-blend'),

    // Feature Sliders
    chkDb: document.getElementById('chk-db'),
    sliderDbStrength: document.getElementById('slider-db-strength'),
    valDbStrength: document.getElementById('val-db-strength'),

    chkSmooth: document.getElementById('chk-smooth'),
    sliderSmoothStrength: document.getElementById('slider-smooth-strength'),
    valSmoothStrength: document.getElementById('val-smooth-strength'),
    sliderTextureKeep: document.getElementById('slider-texture-keep'),
    valTextureKeep: document.getElementById('val-texture-keep'),

    chkLighten: document.getElementById('chk-lighten'),
    sliderLightenStrength: document.getElementById('slider-lighten-strength'),
    valLightenStrength: document.getElementById('val-lighten-strength'),

    chkEyesTeeth: document.getElementById('chk-eyes-teeth'),
    btnDetectTeeth: document.getElementById('btn-detect-teeth'),
    btnDetectTeethText: document.getElementById('btn-detect-teeth-text'),
    sliderTeeth: document.getElementById('slider-teeth'),
    valTeeth: document.getElementById('val-teeth'),
    sliderEyes: document.getElementById('slider-eyes'),
    valEyes: document.getElementById('val-eyes'),

    chkShine: document.getElementById('chk-shine'),
    sliderShine: document.getElementById('slider-shine'),
    valShine: document.getElementById('val-shine')
  };

  // --- INITIALIZATION ---
  async function init() {
    applyTheme(state.theme);
    setupEventListeners();
    setupDropzone();
    setupCanvasInteractions();
    checkServerHealth();
    setInterval(checkServerHealth, 15000);
    if (window.lucide) window.lucide.createIcons();
  }

  // --- THEME SWITCHER (EXTENSIBLE MULTI-THEME ENGINE) ---
  function applyTheme(theme) {
    state.theme = theme || 'dark';
    localStorage.setItem('ai_studio_theme', state.theme);
    document.documentElement.setAttribute('data-theme', state.theme);
    document.documentElement.classList.toggle('light', state.theme === 'light');
    document.documentElement.classList.toggle('dark', state.theme !== 'light');
    if (elements.selectThemePalette) {
      elements.selectThemePalette.value = state.theme;
    }
    if (window.lucide) window.lucide.createIcons();
  }

  function toggleTheme() {
    const newTheme = state.theme === 'dark' ? 'light' : 'dark';
    applyTheme(newTheme);
  }

  // --- CUSTOM TOAST NOTIFICATIONS (ZERO NATIVE ALERTS) ---
  function showToast(message, type = 'info', durationMs = 3200) {
    if (!elements.toastContainer) return;

    const iconMap = {
      success: { icon: 'check-circle-2', color: 'text-emerald-400', border: 'border-emerald-500/30' },
      warning: { icon: 'alert-triangle', color: 'text-amber-400', border: 'border-amber-500/30' },
      error: { icon: 'alert-circle', color: 'text-rose-400', border: 'border-rose-500/30' },
      info: { icon: 'info', color: 'text-blue-400', border: 'border-blue-500/30' }
    };
    const conf = iconMap[type] || iconMap.info;

    const toast = document.createElement('div');
    toast.className = `studio-toast ${conf.border}`;
    toast.innerHTML = `
      <i data-lucide="${conf.icon}" class="w-4 h-4 ${conf.color} shrink-0 mt-0.5"></i>
      <div class="flex-1 text-xs text-slate-200 font-medium leading-tight">${message}</div>
    `;

    elements.toastContainer.appendChild(toast);
    if (window.lucide) window.lucide.createIcons();

    setTimeout(() => {
      toast.classList.add('toast-leaving');
      setTimeout(() => {
        if (toast.parentNode) toast.parentNode.removeChild(toast);
      }, 250);
    }, durationMs);
  }

  // --- CUSTOM MODAL DIALOG (ZERO NATIVE CONFIRM / ALERT) ---
  function showCustomConfirm(title, message, onConfirm, onCancel) {
    if (!elements.modalCustomDialog) return;

    elements.dialogTitle.textContent = title;
    elements.dialogMessage.textContent = message;
    elements.btnDialogCancel.style.display = 'inline-block';
    elements.btnDialogConfirm.textContent = 'Confirm';

    elements.modalCustomDialog.classList.remove('hidden');

    const handleConfirm = () => {
      cleanup();
      if (onConfirm) onConfirm();
    };

    const handleCancel = () => {
      cleanup();
      if (onCancel) onCancel();
    };

    const cleanup = () => {
      elements.btnDialogConfirm.removeEventListener('click', handleConfirm);
      elements.btnDialogCancel.removeEventListener('click', handleCancel);
      elements.modalCustomDialog.classList.add('hidden');
    };

    elements.btnDialogConfirm.addEventListener('click', handleConfirm);
    elements.btnDialogCancel.addEventListener('click', handleCancel);
    if (window.lucide) window.lucide.createIcons();
  }

  function showCustomAlert(title, message, type = 'info') {
    if (!elements.modalCustomDialog) return;

    elements.dialogTitle.textContent = title;
    elements.dialogMessage.textContent = message;
    elements.btnDialogCancel.style.display = 'none';
    elements.btnDialogConfirm.textContent = 'OK';

    elements.modalCustomDialog.classList.remove('hidden');

    const handleOk = () => {
      elements.btnDialogConfirm.removeEventListener('click', handleOk);
      elements.modalCustomDialog.classList.add('hidden');
    };

    elements.btnDialogConfirm.addEventListener('click', handleOk);
    if (window.lucide) window.lucide.createIcons();
  }

  // --- RIBBON TAB SWITCHER ---
  function switchRibbonTab(tabName) {
    state.activeRibbonTab = tabName;

    elements.ribbonTabs.forEach(tab => {
      const isTarget = (tab.getAttribute('data-tab') === tabName);
      tab.classList.toggle('active', isTarget);
    });

    Object.keys(elements.ribbonPanels).forEach(key => {
      const panel = elements.ribbonPanels[key];
      if (panel) {
        panel.classList.toggle('hidden', key !== tabName);
      }
    });

    if (window.lucide) window.lucide.createIcons();
  }

  // --- SERVER HEALTH CHECK ---
  async function checkServerHealth() {
    try {
      const res = await fetch(`${state.serverUrl}/health`, { method: 'GET', signal: AbortSignal.timeout(4000) });
      if (res.ok) {
        const data = await res.json();
        elements.serverStatusDot.className = 'w-1.5 h-1.5 rounded-full bg-emerald-500';
        elements.serverStatusText.textContent = `Server Ready (${data.device || 'AI Engine'})`;

        const hasGemini = data.gemini_enabled || Boolean(state.apiKey);
        elements.geminiStatusText.textContent = hasGemini ? 'Gemini Vision: Active' : 'Gemini Vision: Offline';
      } else {
        throw new Error('Health check failed');
      }
    } catch (e) {
      elements.serverStatusDot.className = 'w-1.5 h-1.5 rounded-full bg-rose-500';
      elements.serverStatusText.textContent = 'Server Disconnected';
    }
  }

  // --- DROPZONE & IMAGE LOADING ---
  function setupDropzone() {
    const dropzone = elements.uploadDropzone;

    dropzone.addEventListener('click', () => elements.fileInput.click());
    if (elements.btnBrowseFile) {
      elements.btnBrowseFile.addEventListener('click', (e) => {
        e.stopPropagation();
        elements.fileInput.click();
      });
    }
    if (elements.btnBrowseFileCenter) {
      elements.btnBrowseFileCenter.addEventListener('click', (e) => {
        e.stopPropagation();
        elements.fileInput.click();
      });
    }

    elements.fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        loadImageFile(e.target.files[0]);
      }
    });

    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.add('border-blue-500', 'bg-blue-500/10');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.remove('border-blue-500', 'bg-blue-500/10');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        loadImageFile(e.dataTransfer.files[0]);
      }
    });
  }

  function loadImageFile(file) {
    if (!file.type.startsWith('image/')) {
      showCustomAlert('Invalid File', 'Please upload a valid photograph (JPEG, PNG, WEBP, or TIFF).', 'warning');
      return;
    }

    state.originalFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        state.originalImage = img;
        state.originalWidth = img.naturalWidth;
        state.originalHeight = img.naturalHeight;

        // Reset state
        state.retouchedImage = img;
        state.blobs = [];
        state.spotHistory = [];
        state.spotHistoryIndex = -1;
        state.selectedBlobIndex = -1;
        state.skinMaskImage = null;
        state.skinMaskBlob = null;
        deselectSpot();

        // Show canvas stage
        elements.uploadDropzone.classList.add('hidden');
        elements.canvasContainer.classList.remove('hidden');
        elements.btnAnalyze.disabled = false;
        if (elements.btnAnalyzeBlemish) elements.btnAnalyzeBlemish.disabled = false;
        elements.btnExportMain.disabled = false;

        elements.labelImageDim.textContent = `${img.naturalWidth} × ${img.naturalHeight} px`;
        updateSpotsCountLabel();
        elements.labelSkinToneContainer.style.display = 'none';

        resizeCanvases();
        fitZoomToScreen();
        renderCanvases();

        showToast('Photo loaded. Open "Blemishes & Healing" and click "Detect Blemishes" to scan.', 'info', 4000);
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  // --- CANVAS SIZING & MATRIX-BASED TRANSFORMS (NO SHIFTING TO RIGHT) ---
  function resizeCanvases() {
    if (!state.originalImage) return;

    const w = state.originalWidth;
    const h = state.originalHeight;

    const canvases = [elements.canvasBefore, elements.canvasAfter, elements.canvasOverlay];
    canvases.forEach(c => {
      c.width = w;
      c.height = h;
    });

    // Keep container fixed to native image dimensions; all scale/pan handled by GPU matrix transform
    elements.canvasContainer.style.width = `${w}px`;
    elements.canvasContainer.style.height = `${h}px`;

    updateCanvasTransforms();
  }

  function updateCanvasTransforms() {
    if (!state.originalImage) return;

    // Pure GPU transform with fixed top-left origin
    elements.canvasContainer.style.transform = `translate3d(${state.panX}px, ${state.panY}px, 0px) scale(${state.zoom})`;

    const p = Math.max(0, Math.min(100, state.splitPercent));
    if (elements.canvasAfter) {
      elements.canvasAfter.style.clipPath = `polygon(0 0, ${p}% 0, ${p}% 100%, 0 100%)`;
      elements.canvasAfter.style.webkitClipPath = `polygon(0 0, ${p}% 0, ${p}% 100%, 0 100%)`;
    }
    if (elements.splitDivider) {
      elements.splitDivider.style.left = `${p}%`;
    }

    elements.labelZoomLevel.textContent = `${Math.round(state.zoom * 100)}%`;
  }

  function fitZoomToScreen() {
    if (!state.originalImage) return;

    const stageW = elements.viewportStage.clientWidth;
    const stageH = elements.viewportStage.clientHeight;

    const scaleX = (stageW - 60) / state.originalWidth;
    const scaleY = (stageH - 60) / state.originalHeight;
    state.zoom = Math.min(1.0, Math.min(scaleX, scaleY));

    // Perfectly center the image in the viewport stage
    state.panX = Math.round((stageW - state.originalWidth * state.zoom) / 2);
    state.panY = Math.round((stageH - state.originalHeight * state.zoom) / 2);

    updateCanvasTransforms();
  }

  function setZoom(newZoom) {
    if (!state.originalImage) return;

    const stageW = elements.viewportStage.clientWidth;
    const stageH = elements.viewportStage.clientHeight;

    // Center focal point on screen center
    const centerX = stageW / 2;
    const centerY = stageH / 2;

    const oldZoom = state.zoom;
    const clamped = Math.max(0.12, Math.min(8.0, newZoom));

    state.panX = centerX - (centerX - state.panX) * (clamped / oldZoom);
    state.panY = centerY - (centerY - state.panY) * (clamped / oldZoom);
    state.zoom = clamped;

    updateCanvasTransforms();
  }

  function getCanvasImageCoords(clientX, clientY) {
    const rect = elements.viewportStage.getBoundingClientRect();
    const mouseStageX = clientX - rect.left;
    const mouseStageY = clientY - rect.top;

    const imgX = (mouseStageX - state.panX) / state.zoom;
    const imgY = (mouseStageY - state.panY) / state.zoom;

    const onCanvas = (imgX >= 0 && imgX <= state.originalWidth && imgY >= 0 && imgY <= state.originalHeight);
    return { imgX, imgY, onCanvas };
  }

  function renderCanvases() {
    if (!state.originalImage) return;

    const w = state.originalWidth;
    const h = state.originalHeight;

    // 1. Draw Before Canvas
    const ctxBefore = elements.canvasBefore.getContext('2d');
    ctxBefore.clearRect(0, 0, w, h);
    ctxBefore.drawImage(state.originalImage, 0, 0, w, h);

    // 2. Draw After Canvas
    const ctxAfter = elements.canvasAfter.getContext('2d');
    ctxAfter.clearRect(0, 0, w, h);
    if (state.retouchedImage) {
      ctxAfter.drawImage(state.retouchedImage, 0, 0, w, h);
    } else {
      ctxAfter.drawImage(state.originalImage, 0, 0, w, h);
    }

    // 3. Draw Overlay Canvas
    renderOverlayCanvas();
  }

  function renderOverlayCanvas() {
    const ctx = elements.canvasOverlay.getContext('2d');
    const w = state.originalWidth;
    const h = state.originalHeight;
    ctx.clearRect(0, 0, w, h);

    // 1. Draw Skin Mask
    if (state.showSkinMask && state.skinMaskImage) {
      ctx.save();
      ctx.globalAlpha = 0.35;
      ctx.drawImage(state.skinMaskImage, 0, 0, w, h);
      ctx.restore();
    }

    // 2. Draw Blemish Spots
    if (state.showSpots && state.blobs && state.blobs.length > 0) {
      state.blobs.forEach((blob, idx) => {
        const [cx, cy] = blob.centroid;
        const radius = Math.max(3, Math.min(22, blob.radius || 6));
        const isActive = blob.active !== false;
        const isHovered = (idx === state.hoveredBlobIndex);
        const isSelected = (idx === state.selectedBlobIndex);

        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);

        if (isSelected) {
          ctx.strokeStyle = '#38bdf8';
          ctx.lineWidth = 3.5;
          ctx.fillStyle = 'rgba(56, 189, 248, 0.40)';
        } else if (isActive) {
          if (isHovered && state.spotToolMode === 'erase') {
            ctx.strokeStyle = '#f43f5e';
            ctx.lineWidth = 3.2;
            ctx.fillStyle = 'rgba(244, 63, 94, 0.50)';
          } else {
            ctx.strokeStyle = isHovered ? '#ffffff' : '#ef4444';
            ctx.lineWidth = isHovered ? 2.8 : 1.8;
            ctx.fillStyle = isHovered ? 'rgba(239, 68, 68, 0.45)' : 'rgba(239, 68, 68, 0.20)';
          }
        } else {
          ctx.setLineDash([3, 3]);
          ctx.strokeStyle = isHovered ? '#ffffff' : 'rgba(148, 163, 184, 0.8)';
          ctx.lineWidth = isHovered ? 2.2 : 1.4;
          ctx.fillStyle = 'rgba(148, 163, 184, 0.12)';
        }

        ctx.fill();
        ctx.stroke();

        // Selected Ring
        if (isSelected) {
          ctx.beginPath();
          ctx.arc(cx, cy, radius + 3, 0, Math.PI * 2);
          ctx.setLineDash([2, 2]);
          ctx.strokeStyle = '#38bdf8';
          ctx.lineWidth = 1.6;
          ctx.stroke();
        }

        // Center dot
        ctx.beginPath();
        ctx.arc(cx, cy, isSelected ? 2.2 : 1.5, 0, Math.PI * 2);
        ctx.fillStyle = isSelected ? '#38bdf8' : (isActive ? '#ef4444' : '#94a3b8');
        ctx.fill();

        // Hover Erase X
        if (isHovered && state.spotToolMode === 'erase') {
          const arm = Math.max(2.5, radius * 0.55);
          ctx.beginPath();
          ctx.moveTo(cx - arm, cy - arm);
          ctx.lineTo(cx + arm, cy + arm);
          ctx.moveTo(cx + arm, cy - arm);
          ctx.lineTo(cx - arm, cy + arm);
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2.0;
          ctx.stroke();
        }

        ctx.restore();
      });
    }

    // 3. Dynamic Brush Indicator in Add Mode
    if (state.activeTool === 'blemish' && state.spotToolMode === 'add' && state.cursorImgPos.onCanvas) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(state.cursorImgPos.x, state.cursorImgPos.y, state.spotBrushRadius, 0, Math.PI * 2);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 1.8;
      ctx.fillStyle = 'rgba(59, 130, 246, 0.22)';
      ctx.fill();
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(state.cursorImgPos.x, state.cursorImgPos.y, 1.5, 0, Math.PI * 2);
      ctx.fillStyle = '#3b82f6';
      ctx.fill();
      ctx.restore();
    }
  }

  // --- SPOT MANAGEMENT & CLIENT-SIDE INSTANT EDITING ---
  function pushSpotHistory() {
    if (state.spotHistoryIndex < state.spotHistory.length - 1) {
      state.spotHistory = state.spotHistory.slice(0, state.spotHistoryIndex + 1);
    }
    state.spotHistory.push(JSON.parse(JSON.stringify(state.blobs)));
    if (state.spotHistory.length > 30) {
      state.spotHistory.shift();
    }
    state.spotHistoryIndex = state.spotHistory.length - 1;
    updateUndoRedoButtons();
  }

  function undoSpotHistory() {
    if (state.spotHistoryIndex > 0) {
      state.spotHistoryIndex--;
      state.blobs = JSON.parse(JSON.stringify(state.spotHistory[state.spotHistoryIndex]));
      state.selectedBlobIndex = -1;
      deselectSpot();
      updateSpotsCountLabel();
      renderOverlayCanvas();
      scheduleLivePreview(60);
      updateUndoRedoButtons();
      showToast('Undo spot action', 'info', 1500);
    }
  }

  function redoSpotHistory() {
    if (state.spotHistoryIndex < state.spotHistory.length - 1) {
      state.spotHistoryIndex++;
      state.blobs = JSON.parse(JSON.stringify(state.spotHistory[state.spotHistoryIndex]));
      state.selectedBlobIndex = -1;
      deselectSpot();
      updateSpotsCountLabel();
      renderOverlayCanvas();
      scheduleLivePreview(60);
      updateUndoRedoButtons();
      showToast('Redo spot action', 'info', 1500);
    }
  }

  function updateUndoRedoButtons() {
    if (elements.btnUndoSpots) {
      elements.btnUndoSpots.disabled = state.spotHistoryIndex <= 0;
      elements.btnUndoSpots.classList.toggle('opacity-30', state.spotHistoryIndex <= 0);
    }
    if (elements.btnRedoSpots) {
      elements.btnRedoSpots.disabled = state.spotHistoryIndex >= state.spotHistory.length - 1;
      elements.btnRedoSpots.classList.toggle('opacity-30', state.spotHistoryIndex >= state.spotHistory.length - 1);
    }
  }

  function updateSpotsCountLabel() {
    const total = state.blobs.length;
    const active = state.blobs.filter(b => b.active !== false).length;
    if (total === 0) {
      elements.labelSpotsCount.textContent = '0 Spots';
    } else if (active === total) {
      elements.labelSpotsCount.textContent = `${total} Spots Detected`;
    } else {
      elements.labelSpotsCount.textContent = `${active} / ${total} Spots Active`;
    }
  }

  function setSpotToolMode(mode) {
    state.spotToolMode = mode;

    const toolBtns = [elements.btnModeToggle, elements.btnModeAdd, elements.btnModeErase];
    const toolModes = ['toggle', 'add', 'erase'];
    toolBtns.forEach((btn, i) => {
      if (!btn) return;
      if (toolModes[i] === mode) {
        btn.className = 'spot-mode-btn px-2 py-1 rounded bg-blue-600 text-white text-[11px] font-medium shadow-sm transition flex items-center gap-1';
      } else {
        btn.className = 'spot-mode-btn px-2 py-1 rounded text-[11px] font-medium text-slate-400 hover:text-white transition flex items-center gap-1';
      }
    });

    renderOverlayCanvas();
  }

  function setSpotRadius(radius) {
    const clamped = Math.max(3, Math.min(18, parseInt(radius, 10) || 6));
    state.spotBrushRadius = clamped;

    if (elements.sliderSpotRadius) elements.sliderSpotRadius.value = clamped;
    if (elements.labelSpotRadiusToolbar) elements.labelSpotRadiusToolbar.textContent = `${clamped}px`;

    if (state.selectedBlobIndex >= 0 && state.blobs[state.selectedBlobIndex]) {
      state.blobs[state.selectedBlobIndex].radius = clamped;
      if (elements.sliderInspectorRadius) elements.sliderInspectorRadius.value = clamped;
      if (elements.valInspectorRadius) elements.valInspectorRadius.textContent = `${clamped} px`;
      pushSpotHistory();
      scheduleLivePreview(50);
    }

    renderOverlayCanvas();
  }

  function findClosestBlob(imgX, imgY, hitPadding = 4) {
    let closestIdx = -1;
    let minDist = Infinity;

    for (let i = 0; i < state.blobs.length; i++) {
      const b = state.blobs[i];
      const dist = Math.hypot(b.centroid[0] - imgX, b.centroid[1] - imgY);
      const hitRadius = Math.max(6, (b.radius || 6) + hitPadding);
      if (dist <= hitRadius && dist < minDist) {
        minDist = dist;
        closestIdx = i;
      }
    }

    return closestIdx;
  }

  function selectSpot(index) {
    if (index < 0 || index >= state.blobs.length) {
      deselectSpot();
      return;
    }

    state.selectedBlobIndex = index;
    const b = state.blobs[index];

    if (elements.spotInspectorCard) {
      elements.spotInspectorCard.classList.remove('hidden');
      elements.labelInspectorSpotId.textContent = `Spot #${b.id || (index + 1)}`;
      elements.labelInspectorCoords.textContent = `X: ${Math.round(b.centroid[0])}, Y: ${Math.round(b.centroid[1])}`;
      elements.sliderInspectorRadius.value = b.radius || 6;
      elements.valInspectorRadius.textContent = `${b.radius || 6} px`;
    }

    if (state.activeRibbonTab !== 'blemish') {
      switchRibbonTab('blemish');
    }

    renderOverlayCanvas();
  }

  function deselectSpot() {
    state.selectedBlobIndex = -1;
    if (elements.spotInspectorCard) {
      elements.spotInspectorCard.classList.add('hidden');
    }
    renderOverlayCanvas();
  }

  function clientAddSpot(x, y, radius = state.spotBrushRadius) {
    const maxId = state.blobs.reduce((max, b) => Math.max(max, b.id || 0), 0);
    const r = Math.max(3, Math.min(18, radius));
    const pad = 1;

    const newBlob = {
      id: maxId + 1,
      bbox: [x - r - pad, y - r - pad, x + r + pad, y + r + pad],
      centroid: [x, y],
      radius: r,
      area: Math.round(Math.PI * r * r),
      confidence: 0.99,
      active: true,
      label: 'user_added',
      source: 'click'
    };

    state.blobs.push(newBlob);
    pushSpotHistory();
    updateSpotsCountLabel();
    selectSpot(state.blobs.length - 1);
    renderOverlayCanvas();
    scheduleLivePreview(40);
  }

  function clientDeleteSpot(x, y) {
    const closestIdx = findClosestBlob(x, y, 6);

    if (closestIdx >= 0) {
      state.blobs.splice(closestIdx, 1);
      state.hoveredBlobIndex = -1;
      if (state.selectedBlobIndex === closestIdx) {
        deselectSpot();
      } else if (state.selectedBlobIndex > closestIdx) {
        state.selectedBlobIndex--;
      }
      elements.spotTooltip.classList.add('hidden');
      pushSpotHistory();
      updateSpotsCountLabel();
      renderOverlayCanvas();
      scheduleLivePreview(40);
      return true;
    }
    return false;
  }

  function clientToggleSpot(x, y) {
    const closestIdx = findClosestBlob(x, y, 4);

    if (closestIdx >= 0) {
      state.blobs[closestIdx].active = !(state.blobs[closestIdx].active !== false);
      selectSpot(closestIdx);
      pushSpotHistory();
      updateSpotsCountLabel();
      renderOverlayCanvas();
      scheduleLivePreview(40);
    } else {
      clientAddSpot(x, y, state.spotBrushRadius);
    }
  }

  function clientMergeOverlappingSpots() {
    if (state.blobs.length <= 1) return;

    const beforeCount = state.blobs.length;
    const sorted = [...state.blobs].sort((a, b) => (b.confidence || 0.8) - (a.confidence || 0.8));
    const merged = [];

    for (const b of sorted) {
      const [cx1, cy1] = b.centroid;
      const r1 = b.radius || 6;
      let isOverlap = false;

      for (const s of merged) {
        const [cx2, cy2] = s.centroid;
        const r2 = s.radius || 6;
        const dist = Math.hypot(cx1 - cx2, cy1 - cy2);

        if (dist < ((r1 + r2) * 0.65) || dist < Math.min(r1, r2)) {
          s.centroid = [Math.round((cx1 + cx2) / 2), Math.round((cy1 + cy2) / 2)];
          s.radius = Math.max(3, Math.min(18, Math.max(r1, r2)));
          isOverlap = true;
          break;
        }
      }

      if (!isOverlap) {
        merged.push({ ...b, id: merged.length + 1 });
      }
    }

    state.blobs = merged;
    deselectSpot();
    pushSpotHistory();
    updateSpotsCountLabel();
    renderOverlayCanvas();
    scheduleLivePreview(40);

    const diff = beforeCount - merged.length;
    if (diff > 0) {
      showToast(`Merged ${diff} duplicate spots into clean single targets.`, 'success');
    }
  }

  function clientSelectAll() {
    state.blobs.forEach(b => b.active = true);
    pushSpotHistory();
    updateSpotsCountLabel();
    renderOverlayCanvas();
    scheduleLivePreview(40);
    showToast('All spots set to Active', 'info', 1800);
  }

  function clientDeselectAll() {
    state.blobs.forEach(b => b.active = false);
    pushSpotHistory();
    updateSpotsCountLabel();
    renderOverlayCanvas();
    scheduleLivePreview(40);
    showToast('All spots preserved', 'info', 1800);
  }

  function clientClearAll() {
    if (state.blobs.length === 0) return;
    showCustomConfirm('Clear All Blemishes', 'Are you sure you want to remove all detected spots?', () => {
      state.blobs = [];
      state.hoveredBlobIndex = -1;
      deselectSpot();
      elements.spotTooltip.classList.add('hidden');
      pushSpotHistory();
      updateSpotsCountLabel();
      renderOverlayCanvas();
      scheduleLivePreview(40);
      showToast('All spots cleared', 'info');
    });
  }

  // --- CANVAS INTERACTIONS (ROCK-SOLID FOCAL ZOOM & PAN) ---
  function setupCanvasInteractions() {
    const stage = elements.viewportStage;
    const overlay = elements.canvasOverlay;
    const handle = elements.splitHandle;
    const divider = elements.splitDivider;

    function moveSplit(clientX) {
      const coords = getCanvasImageCoords(clientX, 0);
      const percent = Math.max(0, Math.min(100, (coords.imgX / state.originalWidth) * 100));
      state.splitPercent = percent;
      updateCanvasTransforms();
    }

    handle.addEventListener('mousedown', (e) => {
      e.stopPropagation();
      state.isDraggingSplit = true;
    });

    divider.addEventListener('mousedown', (e) => {
      e.stopPropagation();
      state.isDraggingSplit = true;
    });

    handle.addEventListener('touchstart', (e) => {
      e.stopPropagation();
      state.isDraggingSplit = true;
    }, { passive: true });

    window.addEventListener('mousemove', (e) => {
      if (state.isDraggingSplit) {
        moveSplit(e.clientX);
      } else if (state.isPanning) {
        state.panX = e.clientX - state.startPanX;
        state.panY = e.clientY - state.startPanY;
        updateCanvasTransforms();
      }
    });

    window.addEventListener('touchmove', (e) => {
      if (state.isDraggingSplit && e.touches && e.touches[0]) {
        moveSplit(e.touches[0].clientX);
      }
    }, { passive: true });

    window.addEventListener('mouseup', () => {
      state.isDraggingSplit = false;
      state.isPanning = false;
      state.isDraggingErase = false;
    });

    window.addEventListener('touchend', () => {
      state.isDraggingSplit = false;
      state.isPanning = false;
      state.isDraggingErase = false;
    });

    // Spacebar, Middle Click, or Ctrl+Click Drag Pan
    stage.addEventListener('mousedown', (e) => {
      // Pan if middle click (1), or space held, or ctrl held, or clicked on empty stage outside canvas
      const coords = getCanvasImageCoords(e.clientX, e.clientY);
      if (e.button === 1 || (e.button === 0 && (e.spaceKey || e.ctrlKey || !coords.onCanvas))) {
        state.isPanning = true;
        state.startPanX = e.clientX - state.panX;
        state.startPanY = e.clientY - state.panY;
        stage.style.cursor = 'grabbing';
        e.preventDefault();
      }
    });

    window.addEventListener('mouseup', () => {
      state.isDraggingSplit = false;
      state.isPanning = false;
      state.isDraggingErase = false;
      stage.style.cursor = 'default';
    });

    stage.addEventListener('contextmenu', (e) => e.preventDefault());

    // Free Ctrl + Wheel and Direct Wheel Focal-Point Zoom (Exact Centering on Spot)
    stage.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (!state.originalImage) return;

      const rect = stage.getBoundingClientRect();
      const mouseStageX = e.clientX - rect.left;
      const mouseStageY = e.clientY - rect.top;

      const oldZoom = state.zoom;
      const zoomFactor = e.deltaY < 0 ? 1.16 : 0.86;
      const newZoom = Math.max(0.08, Math.min(12.0, oldZoom * zoomFactor));

      // Pinpoint focal zoom: image pixel under cursor stays locked under cursor
      state.panX = mouseStageX - (mouseStageX - state.panX) * (newZoom / oldZoom);
      state.panY = mouseStageY - (mouseStageY - state.panY) * (newZoom / oldZoom);
      state.zoom = newZoom;
      updateCanvasTransforms();
    }, { passive: false });

    // Interactive Overlay Hover & Tracking
    overlay.addEventListener('mousemove', (e) => {
      if (!state.originalImage || state.isDraggingSplit || state.isPanning) return;

      const coords = getCanvasImageCoords(e.clientX, e.clientY);
      state.cursorImgPos = { x: coords.imgX, y: coords.imgY, onCanvas: coords.onCanvas };

      if (state.isDraggingErase && state.spotToolMode === 'erase') {
        clientDeleteSpot(coords.imgX, coords.imgY);
        return;
      }

      const closestIdx = findClosestBlob(coords.imgX, coords.imgY, 4);
      state.hoveredBlobIndex = closestIdx;
      renderOverlayCanvas();

      if (closestIdx >= 0 && state.activeTool === 'blemish') {
        const b = state.blobs[closestIdx];
        const statusText = b.active !== false ? 'Active' : 'Preserved';
        elements.spotTooltip.classList.remove('hidden');
        elements.spotTooltip.style.left = `${e.clientX + 14}px`;
        elements.spotTooltip.style.top = `${e.clientY - 14}px`;

        if (state.spotToolMode === 'erase') {
          elements.spotTooltipText.textContent = `Spot #${b.id || (closestIdx + 1)} — Click or Right-Click to Delete`;
        } else {
          elements.spotTooltipText.textContent = `Spot #${b.id || (closestIdx + 1)} (${statusText}, r=${b.radius || 6}px) — Click: Select/Toggle | Right-Click: Delete`;
        }
      } else {
        elements.spotTooltip.classList.add('hidden');
      }
    });

    overlay.addEventListener('mouseleave', () => {
      state.hoveredBlobIndex = -1;
      state.cursorImgPos.onCanvas = false;
      elements.spotTooltip.classList.add('hidden');
      renderOverlayCanvas();
    });

    overlay.addEventListener('mousedown', (e) => {
      if (e.button === 0 && state.spotToolMode === 'erase') {
        state.isDraggingErase = true;
      }
    });

    // Instant Left Click
    overlay.addEventListener('click', async (e) => {
      if (!state.originalImage || state.isDraggingSplit || state.isPanning) return;

      const coords = getCanvasImageCoords(e.clientX, e.clientY);
      const imgX = Math.round(coords.imgX);
      const imgY = Math.round(coords.imgY);

      if (state.activeTool === 'eyedropper') {
        await sendRefinePoint(imgX, imgY, 'sample_tone');
        return;
      }

      if (e.altKey) {
        clientDeleteSpot(imgX, imgY);
        return;
      }

      if (e.shiftKey) {
        clientAddSpot(imgX, imgY, state.spotBrushRadius);
        return;
      }

      if (state.spotToolMode === 'erase') {
        clientDeleteSpot(imgX, imgY);
      } else if (state.spotToolMode === 'add') {
        clientAddSpot(imgX, imgY, state.spotBrushRadius);
      } else {
        const closestIdx = findClosestBlob(imgX, imgY, 4);
        if (closestIdx >= 0) {
          selectSpot(closestIdx);
          clientToggleSpot(imgX, imgY);
        } else {
          deselectSpot();
          clientAddSpot(imgX, imgY, state.spotBrushRadius);
        }
      }
    });

    // Instant Right Click -> Delete Spot
    overlay.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      if (!state.originalImage || state.isDraggingSplit || state.isPanning) return;

      const coords = getCanvasImageCoords(e.clientX, e.clientY);
      clientDeleteSpot(Math.round(coords.imgX), Math.round(coords.imgY));
    });
  }

  // --- API CALL: ANALYZE PORTRAIT ---
  async function triggerAutoAnalyze() {
    if (!state.originalFile || state.isAnalyzing) return;

    state.isAnalyzing = true;
    if (elements.btnAnalyze) {
      elements.btnAnalyze.disabled = true;
      elements.btnAnalyzeText.textContent = 'Analyzing...';
    }
    if (elements.btnAnalyzeBlemish) {
      elements.btnAnalyzeBlemish.disabled = true;
      elements.btnAnalyzeBlemishText.textContent = 'Scanning...';
    }
    showProcessingBadge('Segmenting Skin & Detecting Spots...');

    const formData = new FormData();
    formData.append('image', state.originalFile);
    formData.append('sensitivity', state.params.sensitivity.toString());
    formData.append('detect_pimples', 'true');
    formData.append('detect_skin', 'true');
    formData.append('include_neck', 'true');
    formData.append('preserve_moles', state.params.preserveMoles ? 'true' : 'false');
    formData.append('preserve_freckles', state.params.preserveFreckles ? 'true' : 'false');
    if (state.apiKey) formData.append('gemini_api_key', state.apiKey);

    try {
      const res = await fetch(`${state.serverUrl}/analyze`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) throw new Error(`Analysis failed with status ${res.status}`);

      const data = await res.json();
      state.blobs = data.blobs || [];
      state.baseToneRGB = data.base_tone_rgb || [210, 170, 150];
      state.baseToneLAB = data.base_tone_lab || [170, 140, 140];

      deselectSpot();
      pushSpotHistory();
      updateSpotsCountLabel();
      elements.labelSkinToneContainer.style.display = 'flex';
      elements.toneSwatchBottom.style.backgroundColor = `rgb(${state.baseToneRGB.join(',')})`;
      elements.labelSkinToneText.textContent = `Tone: ${data.fitzpatrick_type || 'Sampled'}`;

      if (data.skin_mask_base64) {
        const maskImg = new Image();
        maskImg.onload = () => {
          state.skinMaskImage = maskImg;
          renderOverlayCanvas();
        };
        maskImg.src = data.skin_mask_base64;
      }

      renderOverlayCanvas();
      scheduleLivePreview(0);
      showToast(`Detected ${state.blobs.length} facial blemishes.`, 'success');

    } catch (err) {
      console.error('Analyze error:', err);
      showCustomAlert('Analysis Error', `Portrait analysis failed: ${err.message}`, 'error');
    } finally {
      state.isAnalyzing = false;
      if (elements.btnAnalyze) {
        elements.btnAnalyze.disabled = false;
        elements.btnAnalyzeText.textContent = 'AI Portrait Analysis';
      }
      if (elements.btnAnalyzeBlemish) {
        elements.btnAnalyzeBlemish.disabled = false;
        elements.btnAnalyzeBlemishText.textContent = 'Detect Blemishes';
      }
      hideProcessingBadge();
    }
  }

  // --- API CALL: REFINE POINT ---
  async function sendRefinePoint(x, y, actionType) {
    if (!state.originalFile) return;

    showProcessingBadge(actionType === 'sample_tone' ? 'Sampling Skin Tone...' : 'Updating Point...');

    const formData = new FormData();
    formData.append('image', state.originalFile);
    formData.append('x', x.toString());
    formData.append('y', y.toString());
    formData.append('action_type', actionType);
    formData.append('blobs_json', JSON.stringify(state.blobs));
    formData.append('default_radius', state.spotBrushRadius.toString());

    try {
      const res = await fetch(`${state.serverUrl}/refine-point`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) throw new Error(`Refine point failed: ${res.statusText}`);

      const data = await res.json();
      if (data.blobs) {
        state.blobs = data.blobs;
        pushSpotHistory();
        updateSpotsCountLabel();
      }

      if (data.sampled_tone) {
        state.baseToneRGB = data.sampled_tone.rgb;
        state.baseToneLAB = data.sampled_tone.lab;
        elements.toneSwatchBottom.style.backgroundColor = `rgb(${state.baseToneRGB.join(',')})`;
        showToast('Skin tone sampled successfully', 'success', 2000);
      }

      renderOverlayCanvas();
      scheduleLivePreview(50);

    } catch (err) {
      console.error('Refine point error:', err);
    } finally {
      hideProcessingBadge();
    }
  }

  // --- API CALL: AI TEXT REFINEMENT ---
  async function sendTextPrompt(promptText) {
    const text = (promptText || elements.inputPrompt.value || '').trim();
    if (!text || !state.originalFile) return;

    showProcessingBadge(`AI Prompt: "${text}"...`);

    const formData = new FormData();
    formData.append('image', state.originalFile);
    formData.append('prompt', text);
    formData.append('blobs_json', JSON.stringify(state.blobs));
    if (state.apiKey) formData.append('gemini_api_key', state.apiKey);

    try {
      const res = await fetch(`${state.serverUrl}/refine-text`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) throw new Error(`Text refinement failed: ${res.statusText}`);

      const data = await res.json();
      if (data.blobs) {
        state.blobs = data.blobs;
        deselectSpot();
        pushSpotHistory();
        updateSpotsCountLabel();
        renderOverlayCanvas();
        scheduleLivePreview(50);
      }

      if (data.message) {
        showToast(data.message, 'info');
      }

    } catch (err) {
      console.error('Prompt error:', err);
      showCustomAlert('Prompt Error', `AI Prompt processing failed: ${err.message}`, 'error');
    } finally {
      hideProcessingBadge();
    }
  }

  // --- API CALL: LIVE RESULT PREVIEW ---
  function scheduleLivePreview(delayMs = 180) {
    if (state.previewDebounceTimer) clearTimeout(state.previewDebounceTimer);
    state.previewDebounceTimer = setTimeout(triggerLivePreview, delayMs);
  }

  async function triggerLivePreview() {
    if (!state.originalFile || state.isPreviewing) return;

    if (state.previewAbortController) {
      state.previewAbortController.abort();
    }
    state.previewAbortController = new AbortController();

    state.isPreviewing = true;
    showProcessingBadge('Rendering Live AI Retouch...');
    const t0 = performance.now();

    const p = state.params;
    const formData = new FormData();
    formData.append('image', state.originalFile);
    formData.append('blobs_json', JSON.stringify(state.blobs));
    formData.append('include_heal', p.includeHeal ? 'true' : 'false');
    formData.append('heal_mode', p.healMode);
    formData.append('include_db', p.includeDb ? 'true' : 'false');
    formData.append('db_strength', p.dbStrength.toString());
    formData.append('include_smooth', p.includeSmooth ? 'true' : 'false');
    formData.append('smooth_strength', p.smoothStrength.toString());
    formData.append('texture_keep', p.textureKeep.toString());
    formData.append('include_lighten', p.includeLighten ? 'true' : 'false');
    formData.append('strength', p.lightenStrength.toString());
    formData.append('include_eyes_teeth', p.includeEyesTeeth ? 'true' : 'false');
    formData.append('teeth_whiten_strength', p.teethWhiten.toString());
    formData.append('eye_brighten_strength', p.eyeBrighten.toString());
    formData.append('include_shine', p.includeShine ? 'true' : 'false');
    formData.append('shine_strength', p.shineStrength.toString());
    formData.append('texture_blend', p.textureBlend.toString());
    formData.append('feather_radius', p.featherRadius.toString());
    formData.append('grain_intensity', p.grainIntensity.toString());
    formData.append('base_tone_lab', JSON.stringify(state.baseToneLAB));
    formData.append('max_size', '1024');

    try {
      const res = await fetch(`${state.serverUrl}/preview`, {
        method: 'POST',
        body: formData,
        signal: state.previewAbortController.signal
      });

      if (!res.ok) throw new Error(`Preview failed: ${res.statusText}`);

      const blob = await res.blob();
      const imgUrl = URL.createObjectURL(blob);
      const prevImg = new Image();
      prevImg.onload = () => {
        state.retouchedImage = prevImg;
        renderCanvases();
        const latency = Math.round(performance.now() - t0);
        elements.labelRenderTime.textContent = `Latency: ${latency}ms`;
      };
      prevImg.src = imgUrl;

    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Preview error:', err);
      }
    } finally {
      state.isPreviewing = false;
      hideProcessingBadge();
    }
  }

  // --- API CALL: EXPORT MASTER RESOLUTION RESULT ---
  async function exportResult() {
    if (!state.originalFile || state.isExporting) return;

    state.isExporting = true;
    elements.exportBtnLabel.textContent = 'Rendering...';
    showProcessingBadge('Rendering Full Master Resolution Image...');

    const p = state.params;
    const formData = new FormData();
    formData.append('image', state.originalFile);
    formData.append('include_heal', p.includeHeal ? 'true' : 'false');
    formData.append('sensitivity', p.sensitivity.toString());
    formData.append('heal_mode', p.healMode);
    formData.append('texture_blend', p.textureBlend.toString());
    formData.append('include_dodge_burn', p.includeDb ? 'true' : 'false');
    formData.append('db_strength', p.dbStrength.toString());
    formData.append('include_smooth', p.includeSmooth ? 'true' : 'false');
    formData.append('smooth_strength', p.smoothStrength.toString());
    formData.append('texture_keep', p.textureKeep.toString());
    formData.append('include_lighten', p.includeLighten ? 'true' : 'false');
    formData.append('lighten_strength', p.lightenStrength.toString());
    formData.append('include_eye_teeth', p.includeEyesTeeth ? 'true' : 'false');
    formData.append('teeth_whiten', p.teethWhiten.toString());
    formData.append('eye_brighten', p.eyeBrighten.toString());
    formData.append('include_shine', p.includeShine ? 'true' : 'false');
    formData.append('shine_strength', p.shineStrength.toString());
    formData.append('feather_radius', p.featherRadius.toString());
    if (state.apiKey) formData.append('gemini_api_key', state.apiKey);

    try {
      const res = await fetch(`${state.serverUrl}/apply-complete-suite`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) throw new Error(`Export render failed: ${res.statusText}`);

      const blob = await res.blob();
      const downloadUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = downloadUrl;
      const baseName = (state.originalFile.name || 'portrait').replace(/\.[^/.]+$/, '');
      a.download = `${baseName}_retouched_master.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      showToast('Master resolution portrait exported!', 'success');

    } catch (err) {
      console.error('Export error:', err);
      showCustomAlert('Export Error', `Export failed: ${err.message}`, 'error');
    } finally {
      state.isExporting = false;
      elements.exportBtnLabel.textContent = 'Export Master';
      hideProcessingBadge();
    }
  }

  // --- UI EVENT LISTENERS ---
  function setupEventListeners() {
    // Ribbon Tab Switchers
    elements.ribbonTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const tabName = tab.getAttribute('data-tab');
        switchRibbonTab(tabName);
      });
    });

    // Theme toggle
    elements.btnThemeToggle.addEventListener('click', toggleTheme);

    // Toolbar Spot Sub-Mode Selectors
    if (elements.btnModeToggle) elements.btnModeToggle.addEventListener('click', () => setSpotToolMode('toggle'));
    if (elements.btnModeAdd) elements.btnModeAdd.addEventListener('click', () => setSpotToolMode('add'));
    if (elements.btnModeErase) elements.btnModeErase.addEventListener('click', () => setSpotToolMode('erase'));
    if (elements.btnCleanOverlaps) elements.btnCleanOverlaps.addEventListener('click', clientMergeOverlappingSpots);

    // Spot Radius Controls
    if (elements.btnSpotRadiusDec) elements.btnSpotRadiusDec.addEventListener('click', () => setSpotRadius(state.spotBrushRadius - 1));
    if (elements.btnSpotRadiusInc) elements.btnSpotRadiusInc.addEventListener('click', () => setSpotRadius(state.spotBrushRadius + 1));
    if (elements.sliderSpotRadius) {
      elements.sliderSpotRadius.addEventListener('input', (e) => setSpotRadius(e.target.value));
    }

    // Spot Undo / Redo Buttons
    if (elements.btnUndoSpots) elements.btnUndoSpots.addEventListener('click', undoSpotHistory);
    if (elements.btnRedoSpots) elements.btnRedoSpots.addEventListener('click', redoSpotHistory);

    // Spot Batch Buttons
    if (elements.btnSpotsSelectAll) elements.btnSpotsSelectAll.addEventListener('click', clientSelectAll);
    if (elements.btnSpotsDeselectAll) elements.btnSpotsDeselectAll.addEventListener('click', clientDeselectAll);
    if (elements.btnSpotsClearAll) elements.btnSpotsClearAll.addEventListener('click', clientClearAll);

    // Selected Spot Inspector Controls
    if (elements.btnInspectorClose) elements.btnInspectorClose.addEventListener('click', deselectSpot);
    if (elements.sliderInspectorRadius) {
      elements.sliderInspectorRadius.addEventListener('input', (e) => {
        const r = parseInt(e.target.value, 10);
        if (elements.valInspectorRadius) elements.valInspectorRadius.textContent = `${r} px`;
        if (state.selectedBlobIndex >= 0 && state.blobs[state.selectedBlobIndex]) {
          state.blobs[state.selectedBlobIndex].radius = r;
          renderOverlayCanvas();
          scheduleLivePreview(50);
        }
      });
    }
    if (elements.btnInspectorToggle) {
      elements.btnInspectorToggle.addEventListener('click', () => {
        if (state.selectedBlobIndex >= 0 && state.blobs[state.selectedBlobIndex]) {
          const b = state.blobs[state.selectedBlobIndex];
          b.active = !(b.active !== false);
          selectSpot(state.selectedBlobIndex);
          pushSpotHistory();
          updateSpotsCountLabel();
          renderOverlayCanvas();
          scheduleLivePreview(40);
        }
      });
    }
    if (elements.btnInspectorDelete) {
      elements.btnInspectorDelete.addEventListener('click', () => {
        if (state.selectedBlobIndex >= 0 && state.blobs[state.selectedBlobIndex]) {
          state.blobs.splice(state.selectedBlobIndex, 1);
          deselectSpot();
          pushSpotHistory();
          updateSpotsCountLabel();
          renderOverlayCanvas();
          scheduleLivePreview(40);
        }
      });
    }

    // Tool buttons
    if (elements.toolBtnBlemish) elements.toolBtnBlemish.addEventListener('click', () => setTool('blemish'));
    if (elements.toolBtnEyedropper) elements.toolBtnEyedropper.addEventListener('click', () => setTool('eyedropper'));

    // View modes
    if (elements.viewModeSplit) elements.viewModeSplit.addEventListener('click', () => setViewMode('split'));
    if (elements.viewModeAfter) elements.viewModeAfter.addEventListener('click', () => setViewMode('after'));
    if (elements.viewModeBefore) elements.viewModeBefore.addEventListener('click', () => setViewMode('before'));

    // Overlay toggles (View tab + Blemishes tab synchronized)
    function setShowSpots(visible) {
      state.showSpots = Boolean(visible);
      if (elements.chkShowSpots) elements.chkShowSpots.checked = state.showSpots;
      if (elements.btnToggleSpotsBlemish) {
        if (state.showSpots) {
          elements.btnToggleSpotsBlemish.className = 'px-2 py-1 rounded bg-blue-600/20 text-blue-400 border border-blue-500/40 text-[11px] font-medium flex items-center gap-1 hover:bg-blue-600 hover:text-white transition shadow-sm';
        } else {
          elements.btnToggleSpotsBlemish.className = 'px-2 py-1 rounded text-slate-400 hover:text-white text-[11px] font-medium flex items-center gap-1 transition';
        }
      }
      renderOverlayCanvas();
    }

    function setShowSkinMask(visible) {
      state.showSkinMask = Boolean(visible);
      if (elements.chkShowSkinMask) elements.chkShowSkinMask.checked = state.showSkinMask;
      if (elements.btnToggleSkinBlemish) {
        if (state.showSkinMask) {
          elements.btnToggleSkinBlemish.className = 'px-2 py-1 rounded bg-blue-600/20 text-blue-400 border border-blue-500/40 text-[11px] font-medium flex items-center gap-1 hover:bg-blue-600 hover:text-white transition shadow-sm';
        } else {
          elements.btnToggleSkinBlemish.className = 'px-2 py-1 rounded text-slate-400 hover:text-white text-[11px] font-medium flex items-center gap-1 transition';
        }
      }
      renderOverlayCanvas();
    }

    if (elements.chkShowSpots) {
      elements.chkShowSpots.addEventListener('change', (e) => setShowSpots(e.target.checked));
    }
    if (elements.btnToggleSpotsBlemish) {
      elements.btnToggleSpotsBlemish.addEventListener('click', () => setShowSpots(!state.showSpots));
    }
    if (elements.chkShowSkinMask) {
      elements.chkShowSkinMask.addEventListener('change', (e) => setShowSkinMask(e.target.checked));
    }
    if (elements.btnToggleSkinBlemish) {
      elements.btnToggleSkinBlemish.addEventListener('click', () => setShowSkinMask(!state.showSkinMask));
    }

    // Zoom Buttons
    if (elements.btnZoomIn) elements.btnZoomIn.addEventListener('click', () => setZoom(state.zoom * 1.25));
    if (elements.btnZoomOut) elements.btnZoomOut.addEventListener('click', () => setZoom(state.zoom * 0.8));
    if (elements.btnZoomFit) elements.btnZoomFit.addEventListener('click', fitZoomToScreen);

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) redoSpotHistory();
        else undoSpotHistory();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        redoSpotHistory();
      } else if (e.key.toLowerCase() === 't') {
        setSpotToolMode('toggle');
      } else if (e.key.toLowerCase() === 'a') {
        setSpotToolMode('add');
      } else if (e.key.toLowerCase() === 'e') {
        setSpotToolMode('erase');
      } else if (e.key.toLowerCase() === 'm') {
        clientMergeOverlappingSpots();
      } else if (e.key === '[') {
        setSpotRadius(state.spotBrushRadius - 1);
      } else if (e.key === ']') {
        setSpotRadius(state.spotBrushRadius + 1);
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        if (state.selectedBlobIndex >= 0) {
          state.blobs.splice(state.selectedBlobIndex, 1);
          deselectSpot();
          pushSpotHistory();
          updateSpotsCountLabel();
          renderOverlayCanvas();
          scheduleLivePreview(40);
        } else if (state.hoveredBlobIndex >= 0) {
          state.blobs.splice(state.hoveredBlobIndex, 1);
          state.hoveredBlobIndex = -1;
          elements.spotTooltip.classList.add('hidden');
          pushSpotHistory();
          updateSpotsCountLabel();
          renderOverlayCanvas();
          scheduleLivePreview(40);
        }
      } else if (e.key === '1') setTool('blemish');
      else if (e.key === '2') setTool('eyedropper');
      else if (e.key === 'f' || e.key === 'F') fitZoomToScreen();
      else if (e.key === 'Escape') {
        deselectSpot();
        closeSettingsModal();
        closeShortcutsModal();
      }
    });

    // Window resize event for canvas responsiveness
    window.addEventListener('resize', () => {
      if (state.originalImage) {
        fitZoomToScreen();
      }
    });

    // Apply Healing Action
    function handleApplyHealingClick() {
      if (!state.originalFile) {
        showToast('Please open an image first.', 'warning');
        return;
      }
      const activeCount = state.blobs.filter(b => b.active !== false).length;
      if (activeCount === 0) {
        showToast('No active spots marked. Use Add (+) to click on blemishes first.', 'info', 4000);
        return;
      }
      state.params.includeHeal = true;
      showProcessingBadge(`Inpainting ${activeCount} blemishes with Simple-LaMa...`);
      setViewMode('after');
      scheduleLivePreview(0);
      showToast(`Healing ${activeCount} spots with Simple-LaMa...`, 'info', 3000);
    }

    if (elements.btnApplyHealing) elements.btnApplyHealing.addEventListener('click', handleApplyHealingClick);
    if (elements.btnApplyHealingHome) elements.btnApplyHealingHome.addEventListener('click', handleApplyHealingClick);

    // Hero Analyze & Prompts
    if (elements.btnAnalyze) elements.btnAnalyze.addEventListener('click', triggerAutoAnalyze);
    if (elements.btnAnalyzeBlemish) elements.btnAnalyzeBlemish.addEventListener('click', triggerAutoAnalyze);
    elements.btnSendPrompt.addEventListener('click', () => sendTextPrompt());
    elements.inputPrompt.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') sendTextPrompt();
    });

    document.querySelectorAll('.prompt-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const prompt = chip.getAttribute('data-prompt');
        elements.inputPrompt.value = prompt;
        sendTextPrompt(prompt);
      });
    });

    // Preset Selection
    elements.selectPreset.addEventListener('change', (e) => applyPreset(e.target.value));

    // Reset Image
    elements.btnResetImage.addEventListener('click', () => {
      showCustomConfirm('Reset Image', 'Do you want to clear current photo and start with a new one?', () => {
        location.reload();
      });
    });

    // Export
    elements.btnExportMain.addEventListener('click', exportResult);

    // Shortcuts Modal
    elements.btnShortcuts.addEventListener('click', openShortcutsModal);
    elements.btnCloseShortcuts.addEventListener('click', closeShortcutsModal);

    // Settings Modal
    elements.btnSettings.addEventListener('click', openSettingsModal);
    elements.geminiStatusPill.addEventListener('click', openSettingsModal);
    elements.btnCloseSettings.addEventListener('click', closeSettingsModal);
    elements.btnSaveSettings.addEventListener('click', saveSettings);
    if (elements.selectThemePalette) {
      elements.selectThemePalette.addEventListener('change', (e) => applyTheme(e.target.value));
    }

    // Sliders & Feature Toggles
    bindSlider(elements.sliderSensitivity, elements.valSensitivity, '%', v => state.params.sensitivity = v / 100);
    bindSlider(elements.sliderTextureBlend, elements.valTextureBlend, '%', v => state.params.textureBlend = v / 100);
    bindSlider(elements.sliderDbStrength, elements.valDbStrength, '%', v => state.params.dbStrength = v / 100);
    bindSlider(elements.sliderSmoothStrength, elements.valSmoothStrength, '%', v => state.params.smoothStrength = v / 100);
    bindSlider(elements.sliderTextureKeep, elements.valTextureKeep, '%', v => state.params.textureKeep = v / 100);
    bindSlider(elements.sliderLightenStrength, elements.valLightenStrength, '%', v => state.params.lightenStrength = v / 100);
    bindSlider(elements.sliderTeeth, elements.valTeeth, '%', v => state.params.teethWhiten = v / 100);
    bindSlider(elements.sliderEyes, elements.valEyes, '%', v => state.params.eyeBrighten = v / 100);
    bindSlider(elements.sliderShine, elements.valShine, '%', v => state.params.shineStrength = v / 100);

    if (elements.selectHealMode) elements.selectHealMode.addEventListener('change', (e) => { state.params.healMode = e.target.value; scheduleLivePreview(); });
    if (elements.chkPreserveMoles) elements.chkPreserveMoles.addEventListener('change', (e) => { state.params.preserveMoles = e.target.checked; triggerAutoAnalyze(); });
    if (elements.chkPreserveFreckles) elements.chkPreserveFreckles.addEventListener('change', (e) => { state.params.preserveFreckles = e.target.checked; triggerAutoAnalyze(); });

    // Teeth & Eyes Actions
    function triggerDetectAndWhitenTeeth() {
      if (!state.originalImage) {
        showToast('Please upload a photo first.', 'warning');
        return;
      }

      state.params.includeEyesTeeth = true;
      if (elements.chkEyesTeeth) elements.chkEyesTeeth.checked = true;

      state.params.teethWhiten = 0.65;
      if (elements.sliderTeeth) elements.sliderTeeth.value = 65;
      if (elements.valTeeth) elements.valTeeth.textContent = '65%';

      showProcessingBadge('Segmenting Dental Enamel & Whitening...');
      scheduleLivePreview(0);
      showToast('Teeth detected and whitened! Adjust slider to change shade.', 'success', 4000);
    }

    if (elements.btnDetectTeeth) elements.btnDetectTeeth.addEventListener('click', triggerDetectAndWhitenTeeth);

    if (elements.chkDb) elements.chkDb.addEventListener('change', (e) => { state.params.includeDb = e.target.checked; scheduleLivePreview(); });
    if (elements.chkSmooth) elements.chkSmooth.addEventListener('change', (e) => { state.params.includeSmooth = e.target.checked; scheduleLivePreview(); });
    if (elements.chkLighten) elements.chkLighten.addEventListener('change', (e) => { state.params.includeLighten = e.target.checked; scheduleLivePreview(); });
    if (elements.chkEyesTeeth) elements.chkEyesTeeth.addEventListener('change', (e) => { state.params.includeEyesTeeth = e.target.checked; scheduleLivePreview(); });
    if (elements.chkShine) elements.chkShine.addEventListener('change', (e) => { state.params.includeShine = e.target.checked; scheduleLivePreview(); });
  }

  function bindSlider(slider, labelEl, unit, onValueChange) {
    if (!slider) return;
    slider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      if (labelEl) labelEl.textContent = `${Math.round(val)}${unit}`;
      onValueChange(val);
      scheduleLivePreview();
    });
  }

  function setTool(tool) {
    state.activeTool = tool;
    if (tool === 'blemish') {
      if (elements.toolBtnBlemish) elements.toolBtnBlemish.className = 'tool-btn px-2.5 py-1 rounded bg-blue-600 text-white text-[11px] font-medium flex items-center gap-1 shadow-sm';
      if (elements.toolBtnEyedropper) elements.toolBtnEyedropper.className = 'tool-btn flex flex-col items-center justify-center px-2 py-1 rounded hover:bg-slate-100 dark:hover:bg-white/5 text-slate-700 dark:text-slate-200 transition';
    } else {
      if (elements.toolBtnEyedropper) elements.toolBtnEyedropper.className = 'tool-btn flex flex-col items-center justify-center px-2 py-1 rounded bg-blue-600 text-white transition';
      if (elements.toolBtnBlemish) elements.toolBtnBlemish.className = 'tool-btn px-2.5 py-1 rounded text-slate-700 dark:text-slate-300 text-[11px] font-medium flex items-center gap-1 hover:bg-slate-100 dark:hover:bg-white/5 transition';
    }
    renderOverlayCanvas();
  }

  function setViewMode(mode) {
    state.viewMode = mode;
    [elements.viewModeSplit, elements.viewModeAfter, elements.viewModeBefore].forEach(b => {
      if (b) b.className = 'px-2.5 py-1 rounded text-slate-400 hover:text-white transition flex items-center gap-1';
    });

    if (mode === 'split') {
      if (elements.viewModeSplit) elements.viewModeSplit.className = 'px-2.5 py-1 rounded bg-blue-600 text-white font-medium shadow-sm transition flex items-center gap-1';
      state.splitPercent = 50.0;
      if (elements.splitDivider) elements.splitDivider.style.display = 'block';
    } else if (mode === 'after') {
      if (elements.viewModeAfter) elements.viewModeAfter.className = 'px-2.5 py-1 rounded bg-blue-600 text-white font-medium shadow-sm transition flex items-center gap-1';
      state.splitPercent = 100.0;
      if (elements.splitDivider) elements.splitDivider.style.display = 'none';
    } else {
      if (elements.viewModeBefore) elements.viewModeBefore.className = 'px-2.5 py-1 rounded bg-blue-600 text-white font-medium shadow-sm transition flex items-center gap-1';
      state.splitPercent = 0.0;
      if (elements.splitDivider) elements.splitDivider.style.display = 'none';
    }
    updateCanvasTransforms();
  }

  function applyPreset(preset) {
    if (preset === 'natural') {
      setSlider(elements.sliderSmoothStrength, elements.valSmoothStrength, 30, '%', v => state.params.smoothStrength = v / 100);
      setSlider(elements.sliderTextureKeep, elements.valTextureKeep, 65, '%', v => state.params.textureKeep = v / 100);
      setSlider(elements.sliderDbStrength, elements.valDbStrength, 25, '%', v => state.params.dbStrength = v / 100);
      setSlider(elements.sliderLightenStrength, elements.valLightenStrength, 20, '%', v => state.params.lightenStrength = v / 100);
    } else if (preset === 'commercial') {
      setSlider(elements.sliderSmoothStrength, elements.valSmoothStrength, 50, '%', v => state.params.smoothStrength = v / 100);
      setSlider(elements.sliderTextureKeep, elements.valTextureKeep, 40, '%', v => state.params.textureKeep = v / 100);
      setSlider(elements.sliderDbStrength, elements.valDbStrength, 50, '%', v => state.params.dbStrength = v / 100);
      setSlider(elements.sliderLightenStrength, elements.valLightenStrength, 35, '%', v => state.params.lightenStrength = v / 100);
    } else if (preset === 'matte') {
      setSlider(elements.sliderSmoothStrength, elements.valSmoothStrength, 40, '%', v => state.params.smoothStrength = v / 100);
      setSlider(elements.sliderTextureKeep, elements.valTextureKeep, 50, '%', v => state.params.textureKeep = v / 100);
      setSlider(elements.sliderShine, elements.valShine, 75, '%', v => state.params.shineStrength = v / 100);
      if (elements.chkShine) elements.chkShine.checked = true;
      state.params.includeShine = true;
    }
    scheduleLivePreview();
    showToast(`Applied "${preset}" preset`, 'info', 2000);
  }

  function setSlider(slider, labelEl, val, unit, setter) {
    if (!slider) return;
    slider.value = val;
    if (labelEl) labelEl.textContent = `${val}${unit}`;
    setter(val);
  }

  // --- MODALS ---
  function openSettingsModal() {
    if (elements.inputApiKey) elements.inputApiKey.value = state.apiKey;
    if (elements.inputServerUrl) elements.inputServerUrl.value = state.serverUrl;
    if (elements.selectThemePalette) elements.selectThemePalette.value = state.theme;
    if (elements.modalSettings) elements.modalSettings.classList.remove('hidden');
    if (window.lucide) window.lucide.createIcons();
  }

  function closeSettingsModal() {
    if (elements.modalSettings) elements.modalSettings.classList.add('hidden');
  }

  function openShortcutsModal() {
    elements.modalShortcuts.classList.remove('hidden');
    if (window.lucide) window.lucide.createIcons();
  }

  function closeShortcutsModal() {
    elements.modalShortcuts.classList.add('hidden');
  }

  async function saveSettings() {
    state.apiKey = elements.inputApiKey.value.trim() || DEFAULT_KEY;
    state.serverUrl = elements.inputServerUrl.value.trim() || 'http://127.0.0.1:8765';

    localStorage.setItem('ai_gemini_key', state.apiKey);
    localStorage.setItem('ai_server_url', state.serverUrl);
    if (elements.selectThemePalette) {
      applyTheme(elements.selectThemePalette.value);
    }

    if (state.apiKey) {
      try {
        const formData = new FormData();
        formData.append('gemini_api_key', state.apiKey);
        await fetch(`${state.serverUrl}/set-api-key`, { method: 'POST', body: formData });
      } catch (e) {
        console.warn('Failed to sync key to server:', e);
      }
    }

    closeSettingsModal();
    checkServerHealth();
    showToast('Settings saved successfully', 'success');
  }

  // --- HELPERS ---
  function showProcessingBadge(text) {
    elements.processingBadgeText.textContent = text;
    elements.processingBadge.classList.remove('hidden');
  }

  function hideProcessingBadge() {
    elements.processingBadge.classList.add('hidden');
  }

  // Run on DOM ready
  document.addEventListener('DOMContentLoaded', init);

})();
