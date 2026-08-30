/**
 * AI Retouching Photoshop UXP Plugin - v2 Auto-Detection Architecture
 * BiSeNet Skin Segmentation | Hybrid Pimple Detection | Live Result Preview (Before/After) | LaMa Inpainting
 */

const photoshop = require("photoshop");
const { app, core, action } = photoshop;
const uxp = require("uxp");
const { localFileSystem } = uxp.storage;

/* ============================== DOM ============================== */
const statusBadge = document.getElementById("server-status");
const statusLabel = document.getElementById("status-label");
const inputServerUrl = document.getElementById("input-server-url");
const btnRefreshStatus = document.getElementById("btn-refresh-status");

const btnAutoDetect = document.getElementById("btn-auto-detect");
const detectSpinner = document.getElementById("detect-spinner");
const detectBtnText = document.getElementById("detect-btn-text");

const previewCanvas = document.getElementById("preview-canvas");
const canvasWrapper = document.getElementById("canvas-wrapper");
const canvasEmptyOverlay = document.getElementById("canvas-empty-overlay");
const viewImgBefore = document.getElementById("view-img-before");
const viewAfterClip = document.getElementById("view-after-clip");
const viewImgAfter = document.getElementById("view-img-after");
const previewStatusPill = document.getElementById("preview-status");
const blobCounter = document.getElementById("blob-counter");
const toneSwatch = document.getElementById("tone-swatch");

const toolPimple = document.getElementById("tool-pimple");
const toolEyedropper = document.getElementById("tool-eyedropper");
const btnModeBefore = document.getElementById("btn-mode-before");
const btnModeSplit = document.getElementById("btn-mode-split");
const btnModeAfter = document.getElementById("btn-mode-after");
const chkShowPimples = document.getElementById("chk-show-pimples");
const chkShowSkin = document.getElementById("chk-show-skin");

const inputPrompt = document.getElementById("input-prompt");
const btnSendPrompt = document.getElementById("btn-send-prompt");
const chkTrainingMode = document.getElementById("chk-training-mode");
const selectTrainingLabel = document.getElementById("select-training-label");
const selectTrainingSplit = document.getElementById("select-training-split");
const chkTrainingConsent = document.getElementById("chk-training-consent");
const btnExportTraining = document.getElementById("btn-export-training");
const trainingStatus = document.getElementById("training-status");

const chkEnableHeal = document.getElementById("chk-enable-heal");
const chkPreserveMoles = document.getElementById("chk-preserve-moles");
const sliderSensitivity = document.getElementById("slider-sensitivity");
const valSensitivity = document.getElementById("val-sensitivity");
const sliderTexture = document.getElementById("slider-texture");
const valTexture = document.getElementById("val-texture");
const sliderFeather = document.getElementById("slider-feather");
const valFeather = document.getElementById("val-feather");
const sliderGrain = document.getElementById("slider-grain");
const valGrain = document.getElementById("val-grain");

const chkEnableDb = document.getElementById("chk-enable-db");
const sliderDbStrength = document.getElementById("slider-db-strength");
const valDbStrength = document.getElementById("val-db-strength");

const chkEnableSmooth = document.getElementById("chk-enable-smooth");
const sliderSmoothStrength = document.getElementById("slider-smooth-strength");
const valSmoothStrength = document.getElementById("val-smooth-strength");
const sliderTextureKeep = document.getElementById("slider-texture-keep");
const valTextureKeep = document.getElementById("val-texture-keep");

const chkEnableLighten = document.getElementById("chk-enable-lighten");
const sliderStrength = document.getElementById("slider-strength");
const valStrength = document.getElementById("val-strength");
const chkIncludeNeck = document.getElementById("chk-include-neck");

const chkEnableEyesTeeth = document.getElementById("chk-enable-eyes-teeth");
const sliderTeeth = document.getElementById("slider-teeth");
const valTeeth = document.getElementById("val-teeth");
const sliderEyes = document.getElementById("slider-eyes");
const valEyes = document.getElementById("val-eyes");

const chkEnableShine = document.getElementById("chk-enable-shine");
const sliderShine = document.getElementById("slider-shine");
const valShine = document.getElementById("val-shine");

const btnApplyAll = document.getElementById("btn-apply-all");
const applySpinner = document.getElementById("apply-spinner");
const applyBtnText = document.getElementById("apply-btn-text");
const logBox = document.getElementById("log-box");
const logMessage = document.getElementById("log-message");

/* Debug Console Elements */
const btnToggleDebug = document.getElementById("btn-toggle-debug");
const drawerDebug = document.getElementById("drawer-debug");
const debugTerminal = document.getElementById("debug-terminal");
const btnRunDiagnostics = document.getElementById("btn-run-diagnostics");
const btnCopyDebugLogs = document.getElementById("btn-copy-debug-logs");
const btnClearDebugLogs = document.getElementById("btn-clear-debug-logs");

const debugLogs = [];

function logDebug(msg, type = "info") {
  const d = new Date();
  const timeStr = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}.${String(d.getMilliseconds()).padStart(3, '0')}`;
  const formatted = `[${timeStr}] [${type.toUpperCase()}] ${msg}`;
  debugLogs.push(formatted);
  if (debugLogs.length > 300) debugLogs.shift();

  if (debugTerminal) {
    while (debugTerminal.childNodes.length > 50) {
      debugTerminal.removeChild(debugTerminal.firstChild);
    }
    const line = document.createElement("div");
    line.className = `debug-line ${type}`;
    line.textContent = formatted;
    debugTerminal.appendChild(line);
    try {
      debugTerminal.scrollTop = debugTerminal.scrollHeight;
    } catch (e) {}
  }
  console.log(formatted);
}

// Global exception interceptors
window.addEventListener("error", (evt) => {
  logDebug(`Unhandled Error: ${evt.message || evt} (at ${evt.filename}:${evt.lineno})`, "error");
});
window.addEventListener("unhandledrejection", (evt) => {
  logDebug(`Unhandled Promise Rejection: ${evt.reason ? (evt.reason.stack || evt.reason.message || evt.reason) : evt}`, "error");
});

/* ============================== STATE ============================== */
let isServerOnline = false;
let isProcessing = false;
let activeTool = "pimple";          // 'pimple' | 'eyedropper'
let viewMode = "split";             // 'before' | 'split' | 'after'

let currentPortraitBlob = null;     // PNG snapshot of the document
let currentPortraitImage = null;    // Image element for canvas drawing
let portraitDataCanvas = null;      // Offscreen native-res copy for pixel sampling
let currentSkinMaskImage = null;    // Image element (overlay tint)
let currentSkinMaskBlob = null;     // Raw PNG blob (sent back to /preview & /apply-lighten)
let currentBlobs = [];
let sampledBaseTone = { rgb: [230, 180, 150], lab: [185.7, 141.0, 145.0] };

let resultImage = null;             // Rendered AFTER image from /preview
let splitPos = 0.5;
let previewSeq = 0;
let previewTimer = null;
let previewBusy = false;
let previewQueued = false;
let analyzeTimer = null;

let viewGeom = null;                // {scale, offX, offY, renderW, renderH}
let mouseDownInfo = null;
let draggingSplit = false;

/* ============================== HELPERS ============================== */
function setLog(message, type = "info") {
  logMessage.textContent = message;
  logBox.className = `log-box ${type}`;
}

function getServerUrl() {
  return inputServerUrl.value.trim().replace(/\/+$/, "");
}

function setPreviewStatus(state) {
  const labels = {
    idle: "Idle",
    analyzing: "Analyzing\u2026",
    updating: "Updating\u2026",
    live: "Live",
    stale: "Editing\u2026",
    error: "Error"
  };
  previewStatusPill.textContent = labels[state] || state;
  previewStatusPill.className = `preview-pill pill-${state}`;
}

function rgbToHex(r, g, b) {
  return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;
}

function updateToneSwatch(rgb) {
  toneSwatch.style.backgroundColor = rgbToHex(rgb[0], rgb[1], rgb[2]);
}

/** sRGB -> CIE LAB (OpenCV 8-bit convention: L*255/100, a/b shifted +128) */
function rgbToLab(r, g, b) {
  const lin = (c) => {
    c /= 255;
    return c > 0.04045 ? Math.pow((c + 0.055) / 1.055, 2.4) : c / 12.92;
  };
  const R = lin(r), G = lin(g), B = lin(b);
  let X = (R * 0.4124564 + G * 0.3575761 + B * 0.1804375) / 0.95047;
  let Y = (R * 0.2126729 + G * 0.7151522 + B * 0.072175);
  let Z = (R * 0.0193339 + G * 0.119192 + B * 0.9503041) / 1.08883;
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : (7.787 * t) + (16 / 116));
  X = f(X); Y = f(Y); Z = f(Z);
  const Lstar = 116 * Y - 16;
  const astar = 500 * (X - Y);
  const bstar = 200 * (Y - Z);
  return [
    Math.round(Lstar * 255 / 100),
    Math.round(astar + 128),
    Math.round(bstar + 128)
  ];
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  const len = bytes.length;
  const chunkSize = 16384;
  for (let i = 0; i < len; i += chunkSize) {
    const chunk = bytes.subarray(i, Math.min(i + chunkSize, len));
    binary += String.fromCharCode.apply(null, chunk);
  }
  return btoa(binary);
}

async function blobToDataUrl(blob) {
  if (typeof blob === "string") return blob;
  if (blob._dataUrl) return blob._dataUrl;
  const buffer = await blob.arrayBuffer();
  const b64 = arrayBufferToBase64(buffer);
  const mime = blob.type || "image/png";
  const url = `data:${mime};base64,${b64}`;
  blob._dataUrl = url;
  return url;
}

function loadImage(src, holderId = "img-portrait-holder") {
  return new Promise((resolve, reject) => {
    let img = document.getElementById(holderId);
    if (!img) {
      const container = document.getElementById("image-loader-cache") || document.body;
      img = document.createElement("img");
      img.id = holderId;
      container.appendChild(img);
    }
    let settled = false;
    let pollInterval = null;
    let timeoutId = null;

    function finish() {
      if (settled) return;
      const nw = img.naturalWidth || img.width;
      const nh = img.naturalHeight || img.height;
      if (!nw || !nh || nw === 0 || nh === 0) return; // Wait for real pixel dimensions!
      settled = true;
      if (pollInterval) clearInterval(pollInterval);
      if (timeoutId) clearTimeout(timeoutId);
      logDebug(`Image [${holderId}] decoded: ${nw}x${nh}px`, "ok");
      resolve(img);
    }

    img.onload = () => {
      finish();
    };

    img.onerror = (err) => {
      if (settled) return;
      settled = true;
      if (pollInterval) clearInterval(pollInterval);
      if (timeoutId) clearTimeout(timeoutId);
      logDebug(`Image [${holderId}] decode error: ${err ? (err.message || err) : "unknown"}`, "error");
      reject(new Error("Image decode error: " + (err ? (err.message || err) : "unknown")));
    };

    img.src = src;

    // Check if immediately decoded
    if ((img.naturalWidth && img.naturalWidth > 0) || (img.width && img.width > 0)) {
      finish();
      if (settled) return;
    }

    pollInterval = setInterval(() => {
      if (settled) {
        if (pollInterval) clearInterval(pollInterval);
        return;
      }
      if ((img.naturalWidth && img.naturalWidth > 0) || (img.width && img.width > 0)) {
        finish();
      }
    }, 20);

    timeoutId = setTimeout(() => {
      if (pollInterval) clearInterval(pollInterval);
      if (!settled) {
        if ((img.naturalWidth && img.naturalWidth > 0) || (img.width && img.width > 0)) {
          finish();
        } else {
          settled = true;
          logDebug(`Image [${holderId}] decoding timed out.`, "warn");
          resolve(img);
        }
      }
    }, 4000);
  });
}

function dataUrlToBlob(dataUrl, mime = "image/png") {
  const b64 = dataUrl.split(",")[1];
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

/* ============================== SERVER STATUS ============================== */
async function checkServerStatus() {
  const current = getServerUrl();
  const candidateUrls = [
    current,
    "http://127.0.0.1:8765",
    "http://localhost:8765",
    "http://127.0.0.1:8766",
    "http://localhost:8766",
    "http://127.0.0.1:8001"
  ].filter((v, i, a) => v && a.indexOf(v) === i);

  statusBadge.className = "status-badge checking";
  statusLabel.textContent = "Checking\u2026";

  for (const baseUrl of candidateUrls) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3500);
      const response = await fetch(`${baseUrl}/health`, { method: "GET", signal: controller.signal });
      clearTimeout(timeoutId);

      if (response.ok) {
        const data = await response.json();
        isServerOnline = true;
        statusBadge.className = "status-badge online";
        statusLabel.textContent = `Online (${data.device || "AI"})`;
        statusBadge.title = `Model: ${data.model} | Device: ${data.device} | URL: ${baseUrl}`;
        if (inputServerUrl.value.trim().replace(/\/+$/, "") !== baseUrl) inputServerUrl.value = baseUrl;
        return;
      }
    } catch (e) { /* try next */ }
  }

  isServerOnline = false;
  statusBadge.className = "status-badge offline";
  statusLabel.textContent = "Offline";
  statusBadge.title = "Cannot reach AI backend. Run backend\\run_server.bat";
}

checkServerStatus();
setInterval(checkServerStatus, 12000);
btnRefreshStatus.addEventListener("click", checkServerStatus);
inputServerUrl.addEventListener("change", () => {
  currentPortraitBlob = null;
  checkServerStatus();
});

/* ============================== VIEWPORT SIZING ============================== */
function resizeViewport(imgW, imgH) {
  const boxW = Math.max(220, Math.floor(canvasWrapper.clientWidth || 300));
  let cw = boxW;
  let chh = imgW > 0 ? Math.round(cw * (imgH / imgW)) : 240;
  chh = Math.max(170, Math.min(chh, 480));
  previewCanvas.width = cw;
  previewCanvas.height = chh;
}

window.addEventListener("resize", () => {
  if (currentPortraitImage) {
    resizeViewport(currentPortraitImage.naturalWidth || currentPortraitImage.width,
                   currentPortraitImage.naturalHeight || currentPortraitImage.height);
    renderCanvas();
  }
});

/* ============================== CANVAS RENDERING ============================== */
function drawChip(ctx, x, y, text, alignRight = false) {
  if (!ctx) return;
  try {
    ctx.font = "700 9px Segoe UI, sans-serif";
    const padX = 6, padY = 4;
    const tw = ctx.measureText ? ctx.measureText(text).width : (text.length * 5);
    const w = tw + padX * 2;
    const h = 14;
    const rx = alignRight ? x - w : x;
    ctx.fillStyle = "rgba(0,0,0,0.62)";
    ctx.beginPath();
    ctx.rect(rx, y, w, h);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.textBaseline = "middle";
    ctx.fillText(text, rx + padX, y + h / 2 + 0.5);
  } catch (e) {}
}

function drawSplitDivider(ctx, splitX) {
  if (!ctx) return;
  try {
    const ch = previewCanvas.height;
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(splitX, 0);
    ctx.lineTo(splitX, ch);
    ctx.stroke();

    const cy = Math.round(ch / 2);
    ctx.beginPath();
    ctx.arc(splitX, cy, 11, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.strokeStyle = "rgba(59,130,246,0.9)";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = "#1d4ed8";
    ctx.beginPath();
    ctx.moveTo(splitX - 5, cy);
    ctx.lineTo(splitX - 1.5, cy - 3.5);
    ctx.lineTo(splitX - 1.5, cy + 3.5);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(splitX + 5, cy);
    ctx.lineTo(splitX + 1.5, cy - 3.5);
    ctx.lineTo(splitX + 1.5, cy + 3.5);
    ctx.closePath();
    ctx.fill();
  } catch (e) {}
}

function splitCanvasX() {
  if (!viewGeom) return previewCanvas.width / 2;
  return Math.round(viewGeom.offX + viewGeom.renderW * splitPos);
}

function renderCanvas() {
  if (!currentPortraitImage) return;
  const ctx = previewCanvas.getContext("2d");
  const cw = previewCanvas.width;
  const ch = previewCanvas.height;
  if (ctx) ctx.clearRect(0, 0, cw, ch);

  const imgW = currentPortraitImage.naturalWidth || currentPortraitImage.width || 500;
  const imgH = currentPortraitImage.naturalHeight || currentPortraitImage.height || 500;
  const scale = Math.min(cw / imgW, ch / imgH);
  const renderW = Math.round(imgW * scale);
  const renderH = Math.round(imgH * scale);
  const offX = Math.round((cw - renderW) / 2);
  const offY = Math.round((ch - renderH) / 2);
  viewGeom = { scale, offX, offY, renderW, renderH };

  // 1. Position BEFORE Base Image in DOM
  if (viewImgBefore) {
    if (viewImgBefore.src !== currentPortraitImage.src) {
      viewImgBefore.src = currentPortraitImage.src;
    }
    viewImgBefore.style.display = "block";
    viewImgBefore.style.left = `${offX}px`;
    viewImgBefore.style.top = `${offY}px`;
    viewImgBefore.style.width = `${renderW}px`;
    viewImgBefore.style.height = `${renderH}px`;
  }

  // 2. Position and Clip AFTER Image in DOM
  let overlayLimitX = cw;
  if (viewMode === "after") {
    overlayLimitX = -1;
    if (viewAfterClip && viewImgAfter && resultImage) {
      if (viewImgAfter.src !== resultImage.src) {
        viewImgAfter.src = resultImage.src;
      }
      viewAfterClip.style.display = "block";
      viewAfterClip.style.left = `${offX}px`;
      viewAfterClip.style.top = `${offY}px`;
      viewAfterClip.style.width = `${renderW}px`;
      viewAfterClip.style.height = `${renderH}px`;
      viewImgAfter.style.left = "0px";
      viewImgAfter.style.top = "0px";
      viewImgAfter.style.width = `${renderW}px`;
      viewImgAfter.style.height = `${renderH}px`;
    }
  } else if (viewMode === "split") {
    const splitX = splitCanvasX();
    overlayLimitX = splitX;
    if (viewAfterClip && viewImgAfter && resultImage) {
      if (viewImgAfter.src !== resultImage.src) {
        viewImgAfter.src = resultImage.src;
      }
      const splitPx = Math.round(renderW * splitPos);
      viewAfterClip.style.display = "block";
      viewAfterClip.style.left = `${offX + splitPx}px`;
      viewAfterClip.style.top = `${offY}px`;
      viewAfterClip.style.width = `${Math.max(0, renderW - splitPx)}px`;
      viewAfterClip.style.height = `${renderH}px`;
      viewImgAfter.style.left = `${-splitPx}px`;
      viewImgAfter.style.top = "0px";
      viewImgAfter.style.width = `${renderW}px`;
      viewImgAfter.style.height = `${renderH}px`;
    } else if (viewAfterClip) {
      viewAfterClip.style.display = "none";
    }
  } else {
    // Before mode only
    if (viewAfterClip) viewAfterClip.style.display = "none";
  }

  // 3. Draw Vector Elements on Top Canvas (circles, badges, divider line)
  if (!ctx) return;

  if (overlayLimitX > 0 && chkShowPimples && chkShowPimples.checked && currentBlobs.length > 0) {
    for (const blob of currentBlobs) {
      const bcx = blob.centroid[0] * scale + offX;
      const bcy = blob.centroid[1] * scale + offY;
      const br = Math.max(3, blob.radius * scale);
      if (bcx > overlayLimitX + br) continue;

      try {
        if (blob.active !== false) {
          ctx.beginPath();
          ctx.arc(bcx, bcy, br + 2, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(244,63,94,0.22)";
          ctx.fill();
          ctx.beginPath();
          ctx.arc(bcx, bcy, br, 0, Math.PI * 2);
          ctx.strokeStyle = "#f43f5e";
          ctx.lineWidth = 1.6;
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(bcx, bcy, 1.4, 0, Math.PI * 2);
          ctx.fillStyle = "#ffffff";
          ctx.fill();
        } else {
          ctx.beginPath();
          ctx.arc(bcx, bcy, br, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(161,161,170,0.55)";
          ctx.lineWidth = 1.1;
          ctx.stroke();
        }
      } catch (blobDrawErr) {}
    }
  }

  // 4. Split UI Chrome
  if (viewMode === "split") {
    const splitX = splitCanvasX();
    if (splitX > 52) drawChip(ctx, offX + 8, offY + 8, "BEFORE");
    if (cw - splitX > 58 && resultImage) drawChip(ctx, offX + renderW - 8, offY + 8, "AFTER", true);
    drawSplitDivider(ctx, splitX);
  } else if (viewMode === "after") {
    drawChip(ctx, offX + renderW - 8, offY + 8, "AFTER", true);
  } else {
    drawChip(ctx, offX + 8, offY + 8, "BEFORE");
  }

  const activeCount = currentBlobs.filter(b => b.active !== false).length;
  if (blobCounter) blobCounter.textContent = `${activeCount} Spots`;
}

/* ============================== VIEWPORT INTERACTION ============================== */
function canvasPoint(e) {
  const rect = previewCanvas.getBoundingClientRect();
  return [e.clientX - rect.left, e.clientY - rect.top];
}

function imagePoint(px, py) {
  if (!viewGeom) return null;
  return [(px - viewGeom.offX) / viewGeom.scale, (py - viewGeom.offY) / viewGeom.scale];
}

previewCanvas.addEventListener("mousedown", (e) => {
  if (!currentPortraitImage) return;
  const [px, py] = canvasPoint(e);
  mouseDownInfo = { px, py, moved: false };
  if (viewMode === "split" && Math.abs(px - splitCanvasX()) <= 12) {
    draggingSplit = true;
    previewCanvas.style.cursor = "ew-resize";
  }
});

previewCanvas.addEventListener("mousemove", (e) => {
  if (!currentPortraitImage) return;
  const [px, py] = canvasPoint(e);

  if (draggingSplit && viewGeom) {
    splitPos = Math.max(0.03, Math.min(0.97, (px - viewGeom.offX) / viewGeom.renderW));
    renderCanvas();
    return;
  }
  if (mouseDownInfo) {
    if (Math.hypot(px - mouseDownInfo.px, py - mouseDownInfo.py) > 4) mouseDownInfo.moved = true;
    return;
  }
  const nearSplit = viewMode === "split" && Math.abs(px - splitCanvasX()) <= 12;
  previewCanvas.style.cursor = nearSplit ? "ew-resize" : (activeTool === "eyedropper" ? "cell" : "crosshair");
});

previewCanvas.addEventListener("mouseup", (e) => {
  if (draggingSplit) {
    draggingSplit = false;
    mouseDownInfo = null;
    return;
  }
  if (!mouseDownInfo || mouseDownInfo.moved || !viewGeom) {
    mouseDownInfo = null;
    return;
  }
  const [px, py] = canvasPoint(e);
  const pt = imagePoint(px, py);
  mouseDownInfo = null;
  if (!pt) return;
  handleToolClick(pt[0], pt[1], px, py);
});

previewCanvas.addEventListener("mouseleave", () => {
  draggingSplit = false;
  mouseDownInfo = null;
});

function handleToolClick(origX, origY, px, py) {
  const imgW = currentPortraitImage.naturalWidth || currentPortraitImage.width;
  const imgH = currentPortraitImage.naturalHeight || currentPortraitImage.height;
  origX = Math.max(0, Math.min(imgW - 1, Math.round(origX)));
  origY = Math.max(0, Math.min(imgH - 1, Math.round(origY)));

  if (activeTool === "eyedropper") {
    if (!portraitDataCanvas) return;
    const pctx = portraitDataCanvas.getContext("2d");
    const d = pctx.getImageData(Math.floor(origX), Math.floor(origY), 1, 1).data;
    const rgb = [d[0], d[1], d[2]];
    sampledBaseTone.rgb = rgb;
    sampledBaseTone.lab = rgbToLab(rgb[0], rgb[1], rgb[2]);
    updateToneSwatch(rgb);
    setLog(`Base tone sampled: ${rgbToHex(rgb[0], rgb[1], rgb[2])}. Preview updating\u2026`, "success");
    schedulePreview(350);
    return;
  }

  // Blemish tool: toggle nearest blob, or add a new one.
  const scale = viewGeom ? viewGeom.scale : 1;
  let hitIndex = -1;
  let minDist = Infinity;
  for (let i = 0; i < currentBlobs.length; i++) {
    const b = currentBlobs[i];
    const bcx = b.centroid[0] * scale + (viewGeom ? viewGeom.offX : 0);
    const bcy = b.centroid[1] * scale + (viewGeom ? viewGeom.offY : 0);
    const dist = Math.hypot(px - bcx, py - bcy);
    const effectiveRadius = Math.max(10, b.radius * scale + 6);
    if (dist <= effectiveRadius && dist < minDist) {
      minDist = dist;
      hitIndex = i;
    }
  }

  const trainingMode = chkTrainingMode.checked;
  const trainingLabel = selectTrainingLabel.value;

  if (trainingMode && hitIndex !== -1) {
    const blob = currentBlobs[hitIndex];
    blob.training_label = trainingLabel;
    blob.active = trainingLabel === "heal_blemish" || trainingLabel === "tone_irregularity";
    setLog(`Training label: spot #${blob.id} -> ${trainingLabel.replace("_", " ")}.`, "info");
  } else if (trainingMode && trainingLabel === "exclude") {
    setLog("Nothing added: Exclude is only for a detected false positive.", "info");
  } else if (hitIndex !== -1) {
    currentBlobs[hitIndex].active = !currentBlobs[hitIndex].active;
    setLog(currentBlobs[hitIndex].active
      ? `Spot #${currentBlobs[hitIndex].id} will be removed.`
      : `Spot #${currentBlobs[hitIndex].id} ignored.`, "info");
  } else {
    const newId = currentBlobs.length > 0 ? Math.max(...currentBlobs.map(b => b.id)) + 1 : 1;
    const radius = Math.max(8, Math.round(imgW / 140));
    currentBlobs.push({
      id: newId,
      centroid: [origX, origY],
      bbox: [Math.max(0, origX - radius), Math.max(0, origY - radius), Math.min(imgW, origX + radius), Math.min(imgH, origY + radius)],
      radius: radius,
      confidence: 1.0,
      active: !trainingMode || trainingLabel === "heal_blemish" || trainingLabel === "tone_irregularity",
      label: trainingMode ? trainingLabel : "pimple",
      training_label: trainingMode ? trainingLabel : undefined,
      source: "manual_click"
    });
    setLog(trainingMode
      ? `Training annotation added at (${origX}, ${origY}): ${trainingLabel.replace("_", " ")}.`
      : `Missed spot added at (${origX}, ${origY}).`, "success");
  }

  renderCanvas();
  schedulePreview(400);
}

/* Tool + view mode switching */
toolPimple.addEventListener("click", () => {
  activeTool = "pimple";
  toolPimple.classList.add("active");
  toolEyedropper.classList.remove("active");
});

toolEyedropper.addEventListener("click", () => {
  activeTool = "eyedropper";
  toolEyedropper.classList.add("active");
  toolPimple.classList.remove("active");
});

function setViewMode(mode) {
  viewMode = mode;
  btnModeBefore.classList.toggle("active", mode === "before");
  btnModeSplit.classList.toggle("active", mode === "split");
  btnModeAfter.classList.toggle("active", mode === "after");
  renderCanvas();
}
btnModeBefore.addEventListener("click", () => setViewMode("before"));
btnModeSplit.addEventListener("click", () => setViewMode("split"));
btnModeAfter.addEventListener("click", () => setViewMode("after"));

chkShowPimples.addEventListener("change", renderCanvas);
chkShowSkin.addEventListener("change", renderCanvas);

/* ============================== LIVE RESULT PREVIEW (Layer 3) ============================== */
function schedulePreview(delay = 600) {
  if (!currentPortraitBlob) return;
  setPreviewStatus("stale");
  clearTimeout(previewTimer);
  previewTimer = setTimeout(runPreview, delay);
}

async function runPreview() {
  if (!currentPortraitBlob || !isServerOnline) return;
  if (previewBusy) { previewQueued = true; return; }

  const seq = ++previewSeq;
  previewBusy = true;
  setPreviewStatus("updating");

  try {
    const formData = new FormData();
    formData.append("image", currentPortraitBlob, "portrait.png");
    formData.append("blobs_json", JSON.stringify(currentBlobs));
    if (currentSkinMaskBlob) formData.append("skin_mask", currentSkinMaskBlob, "skin_mask.png");
    formData.append("include_heal", chkEnableHeal.checked ? "true" : "false");
    formData.append("include_db", chkEnableDb.checked ? "true" : "false");
    formData.append("db_strength", (parseInt(sliderDbStrength.value, 10) / 100).toString());
    formData.append("include_smooth", chkEnableSmooth.checked ? "true" : "false");
    formData.append("include_lighten", chkEnableLighten.checked ? "true" : "false");
    formData.append("include_eyes_teeth", chkEnableEyesTeeth.checked ? "true" : "false");
    formData.append("teeth_whiten_strength", (parseInt(sliderTeeth.value, 10) / 100).toString());
    formData.append("eye_brighten_strength", (parseInt(sliderEyes.value, 10) / 100).toString());
    formData.append("include_shine", chkEnableShine.checked ? "true" : "false");
    formData.append("shine_strength", (parseInt(sliderShine.value, 10) / 100).toString());
    formData.append("smooth_strength", (parseInt(sliderSmoothStrength.value, 10) / 100).toString());
    formData.append("texture_keep", (parseInt(sliderTextureKeep.value, 10) / 100).toString());
    formData.append("strength", (parseInt(sliderStrength.value, 10) / 100).toString());
    formData.append("texture_blend", (parseInt(sliderTexture.value, 10) / 100).toString());
    formData.append("feather_radius", sliderFeather.value);
    formData.append("grain_intensity", (parseInt(sliderGrain.value, 10) / 100).toString());
    formData.append("base_tone_lab", JSON.stringify(sampledBaseTone.lab));
    formData.append("max_size", "720");

    const res = await fetch(`${getServerUrl()}/preview`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(`Preview failed (${res.status}): ${await res.text()}`);

    const blob = await res.blob();
    const imgUrl = await blobToDataUrl(blob);
    const img = await loadImage(imgUrl, "img-result-holder");
    if (seq !== previewSeq) return; // a newer request superseded this one

    resultImage = img;
    setPreviewStatus("live");
    renderCanvas();

  } catch (err) {
    console.error("Preview error:", err);
    if (seq === previewSeq) {
      setPreviewStatus("error");
      setLog(err.message, "error");
    }
  } finally {
    previewBusy = false;
    if (previewQueued) {
      previewQueued = false;
      schedulePreview(50);
    }
  }
}

/* ============================== PHOTOSHOP EXPORT SNAPSHOT ============================== */
async function exportFullPortrait() {
  logDebug("Starting exportFullPortrait()...", "info");
  const doc = app.activeDocument;
  if (!doc) {
    logDebug("exportFullPortrait failed: No active document found.", "error");
    throw new Error("No active document. Open a portrait photo in Photoshop first.");
  }
  logDebug(`Active document: "${doc.name || 'Untitled'}" (${doc.width}x${doc.height}px, mode=${doc.mode})`, "info");

  // Strategy 1: Ultra-fast native UXP Imaging API
  if (photoshop.imaging && typeof photoshop.imaging.getPixels === "function" && typeof photoshop.imaging.encodeImageData === "function") {
    try {
      logDebug("Capturing via photoshop.imaging API...", "info");
      const t0 = Date.now();
      const base64Data = await core.executeAsModal(async () => {
        const pixels = await photoshop.imaging.getPixels({
          documentID: doc.id
        });
        const encoded = await photoshop.imaging.encodeImageData({
          imageData: pixels.imageData,
          base64: true,
          format: "png"
        });
        pixels.imageData.dispose();
        return encoded;
      }, { commandName: "Capture Portrait Snapshot" });

      if (base64Data) {
        const dt = Date.now() - t0;
        const fullDataUrl = typeof base64Data === "string" && base64Data.startsWith("data:") ? base64Data : `data:image/png;base64,${base64Data}`;
        const blob = dataUrlToBlob(fullDataUrl);
        logDebug(`Imaging capture succeeded in ${dt}ms (${(blob.size / 1024).toFixed(1)} KB)`, "ok");
        return blob;
      }
    } catch (imagingErr) {
      logDebug(`Imaging API fallback: ${imagingErr.message || imagingErr}`, "warn");
    }
  }

  // Strategy 2: Document Save Copy Fallback
  logDebug("Capturing via temporary PNG file export...", "info");
  const t0 = Date.now();
  const tempFolder = await localFileSystem.getTemporaryFolder();
  const portraitFile = await tempFolder.createFile("portrait_temp.png", { overwrite: true });
  const portraitToken = localFileSystem.createSessionToken(portraitFile);

  await core.executeAsModal(async () => {
    await action.batchPlay([
      {
        _obj: "save",
        as: {
          _obj: "PNGFormat",
          method: { _enum: "PNGMethod", _value: "quick" }
        },
        in: { _path: portraitToken, _kind: "local" },
        copy: true,
        lowerCase: true,
        _options: { dialogOptions: "dontDisplay" }
      }
    ], {});
  }, { commandName: "Export Portrait Snapshot" });

  const rawBytes = await portraitFile.read({ format: uxp.storage.formats.binary });
  const b64 = arrayBufferToBase64(rawBytes);
  const blob = new Blob([rawBytes], { type: "image/png" });
  blob._dataUrl = `data:image/png;base64,${b64}`;
  const dt = Date.now() - t0;
  logDebug(`File export succeeded in ${dt}ms (${(blob.size / 1024).toFixed(1)} KB)`, "ok");
  return blob;
}

/* ============================== LAYER PLACEMENT HELPERS ============================== */
async function placePatchAsLayer(pngBuffer, fileName, layerName) {
  const tempFolder = await localFileSystem.getTemporaryFolder();
  const patchFile = await tempFolder.createFile(fileName, { overwrite: true });
  await patchFile.write(pngBuffer, { format: uxp.storage.formats.binary });
  const patchToken = localFileSystem.createSessionToken(patchFile);

  await core.executeAsModal(async () => {
    await action.batchPlay([
      {
        _obj: "placeEvent",
        target: { _path: patchToken, _kind: "local" },
        _options: { dialogOptions: "dontDisplay" }
      }
    ], {});
    const placed = app.activeDocument.activeLayers[0];
    placed.name = layerName;
  }, { commandName: `Place ${layerName}` });
}

async function loadSkinSelection(targetLayerName = null) {
  if (!currentSkinMaskBlob) throw new Error("Analyze Portrait first so Photoshop can build the skin mask.");

  const tempFolder = await localFileSystem.getTemporaryFolder();
  const maskFile = await tempFolder.createFile("ai_skin_selection.png", { overwrite: true });
  await maskFile.write(await currentSkinMaskBlob.arrayBuffer(), { format: uxp.storage.formats.binary });
  const maskToken = localFileSystem.createSessionToken(maskFile);

  await core.executeAsModal(async () => {
    const commands = [
      {
        _obj: "placeEvent",
        target: { _path: maskToken, _kind: "local" },
        _options: { dialogOptions: "dontDisplay" }
      },
      {
        _obj: "set",
        _target: [{ _ref: "channel", _property: "selection" }],
        to: { _ref: "channel", _enum: "channel", _value: "transparencyEnum" },
        _options: { dialogOptions: "dontDisplay" }
      },
      {
        _obj: "delete",
        _target: [{ _ref: "layer", _enum: "ordinal", _value: "targetEnum" }],
        _options: { dialogOptions: "dontDisplay" }
      }
    ];
    if (targetLayerName) {
      commands.push({
        _obj: "select",
        _target: [{ _ref: "layer", _enum: "name", _value: targetLayerName }],
        makeVisible: false,
        _options: { dialogOptions: "dontDisplay" }
      });
    }
    await action.batchPlay(commands, {});
  }, { commandName: "Load AI Skin Mask" });
}

async function attachSkinLayerMask(layerName) {
  await loadSkinSelection(layerName);
  await core.executeAsModal(async () => {
    await action.batchPlay([
      {
        _obj: "make",
        new: { _class: "channel" },
        at: { _ref: "channel", _enum: "channel", _value: "mask" },
        using: { _enum: "userMaskEnabled", _value: "revealSelection" },
        _options: { dialogOptions: "dontDisplay" }
      },
      {
        _obj: "set",
        _target: [{ _ref: "channel", _property: "selection" }],
        to: { _enum: "ordinal", _value: "none" },
        _options: { dialogOptions: "dontDisplay" }
      }
    ], {});
  }, { commandName: `Mask ${layerName}` });
}

function buildToneLiftCurve() {
  const baseL = Math.max(20, Math.min(235, Number(sampledBaseTone.lab[0]) || 160));
  const strength = parseInt(sliderStrength.value, 10) / 100;
  const toneScale = Math.max(0.6, Math.min(1.4, (255 - baseL) / 128));
  const lift = Math.round(strength * 38 * toneScale);
  const points = [
    [0, 0],
    [64, Math.min(255, 64 + Math.round(lift * 0.55))],
    [baseL, Math.min(255, baseL + lift)],
    [192, Math.min(255, 192 + Math.round(lift * 0.35))],
    [255, 255]
  ].sort((a, b) => a[0] - b[0]);

  return points.filter((point, index) => index === 0 || point[0] !== points[index - 1][0])
    .map(([horizontal, vertical]) => ({ _obj: "paint", horizontal, vertical }));
}

async function createMaskedToneLiftLayer() {
  await loadSkinSelection();
  const curve = buildToneLiftCurve();

  await core.executeAsModal(async () => {
    await action.batchPlay([
      {
        _obj: "make",
        _target: [{ _ref: "adjustmentLayer" }],
        using: {
          _obj: "adjustmentLayer",
          name: "AI Skin Tone Lift",
          type: {
            _obj: "curves",
            adjustment: [{
              _obj: "curvesAdjustment",
              channel: { _ref: "channel", _enum: "channel", _value: "composite" },
              curve
            }]
          }
        },
        _options: { dialogOptions: "dontDisplay" }
      },
      {
        _obj: "set",
        _target: [{ _ref: "channel", _property: "selection" }],
        to: { _enum: "ordinal", _value: "none" },
        _options: { dialogOptions: "dontDisplay" }
      }
    ], {});
  }, { commandName: "Create AI Skin Tone Lift" });
}

async function groupRetouchLayers(groupName, layerNames) {
  if (!layerNames || layerNames.length < 2) return;
  try {
    await core.executeAsModal(async () => {
      const selectCommands = [];
      selectCommands.push({
        _obj: "select",
        _target: [{ _ref: "layer", _enum: "name", _value: layerNames[0] }],
        makeVisible: false,
        _options: { dialogOptions: "dontDisplay" }
      });
      for (let i = 1; i < layerNames.length; i++) {
        selectCommands.push({
          _obj: "select",
          _target: [{ _ref: "layer", _enum: "name", _value: layerNames[i] }],
          selectionModifier: { _enum: "selectionModifierType", _value: "addToSelection" },
          makeVisible: false,
          _options: { dialogOptions: "dontDisplay" }
        });
      }
      selectCommands.push({
        _obj: "make",
        _target: [{ _ref: "layerSection" }],
        from: [{ _ref: "layer", _enum: "ordinal", _value: "targetEnum" }],
        name: groupName,
        _options: { dialogOptions: "dontDisplay" }
      });
      await action.batchPlay(selectCommands, {});
    }, { commandName: `Group ${groupName}` });
  } catch (groupErr) {
    logDebug(`Layer grouping note: ${groupErr.message || groupErr}`, "warn");
  }
}

/* ============================== STEP 1: ANALYZE (Layers 1 + 2) ============================== */
async function runAnalysis(isUserInitiated) {
  logDebug(`[ANALYZE] Triggered (isUserInitiated=${isUserInitiated})`, "info");
  if (!isServerOnline) {
    logDebug("[ANALYZE] Server is marked offline, checking connection...", "info");
    await checkServerStatus();
    if (!isServerOnline) {
      logDebug("[ANALYZE] Error: Server offline. Cannot continue.", "error");
      throw new Error("AI backend offline. Start it with backend\\run_server.bat");
    }
  }
  const doc = app.activeDocument;
  if (!doc) {
    logDebug("[ANALYZE] Error: No active document open.", "error");
    throw new Error("No active document. Open a portrait photo in Photoshop first.");
  }

  if (isUserInitiated) setLog("Capturing snapshot \u0026 running AI segmentation\u2026", "info");
  
  logDebug("[ANALYZE] Step 1/4: Capturing document pixels...", "info");
  const portraitBlob = await exportFullPortrait();
  currentPortraitBlob = portraitBlob;

  logDebug("[ANALYZE] Step 2/4: Rendering snapshot to canvas...", "info");
  const imgUrl = await blobToDataUrl(portraitBlob);
  currentPortraitImage = await loadImage(imgUrl);

  const imgW = currentPortraitImage.naturalWidth || currentPortraitImage.width || 500;
  const imgH = currentPortraitImage.naturalHeight || currentPortraitImage.height || 500;

  portraitDataCanvas = document.getElementById("portrait-data-canvas");
  if (portraitDataCanvas) {
    portraitDataCanvas.width = imgW;
    portraitDataCanvas.height = imgH;
    try {
      const pctx = portraitDataCanvas.getContext("2d");
      if (pctx && typeof pctx.drawImage === "function") {
        pctx.drawImage(currentPortraitImage, 0, 0);
      }
    } catch (drawErr) {
      console.warn("Offscreen sampling canvas drawImage:", drawErr);
    }
  }

  resizeViewport(imgW, imgH);
  canvasEmptyOverlay.classList.add("hidden");
  renderCanvas();

  logDebug(`[ANALYZE] Step 3/4: Sending ${imgW}x${imgH}px snapshot to ${getServerUrl()}/analyze...`, "info");
  const formData = new FormData();
  formData.append("image", portraitBlob, "portrait.png");
  formData.append("sensitivity", (parseInt(sliderSensitivity.value, 10) / 100).toString());
  formData.append("detect_pimples", "true");
  formData.append("detect_skin", "true");
  formData.append("include_neck", chkIncludeNeck.checked ? "true" : "false");
  formData.append("preserve_moles", chkPreserveMoles && chkPreserveMoles.checked ? "true" : "false");
  formData.append("feather_radius", sliderFeather ? sliderFeather.value : "3");

  const tStart = Date.now();
  const res = await fetch(`${getServerUrl()}/analyze`, { method: "POST", body: formData });
  const analyzeDuration = Date.now() - tStart;
  
  if (!res.ok) {
    const errText = await res.text();
    logDebug(`[ANALYZE] /analyze returned error ${res.status}: ${errText}`, "error");
    throw new Error(`Analyze failed (${res.status}): ${errText}`);
  }
  const data = await res.json();
  logDebug(`[ANALYZE] Step 4/4: /analyze completed in ${analyzeDuration}ms. Found ${data.blobs ? data.blobs.length : 0} spots, ${data.skin_percentage}% skin.`, "ok");

  // Merge: fresh auto blobs + previously added manual/text blobs that survive
  const previousManual = currentBlobs.filter(b =>
    ["manual_click", "manual", "text_grounding"].includes(b.source || ""));
  const newAuto = data.blobs || [];
  const merged = [...newAuto];
  let nextId = merged.length > 0 ? Math.max(...merged.map(b => b.id)) + 1 : 1;
  for (const mb of previousManual) {
    const covered = newAuto.some(nb =>
      Math.hypot(nb.centroid[0] - mb.centroid[0], nb.centroid[1] - mb.centroid[1]) < (nb.radius * 1.6));
    if (!covered) {
      const copy = Object.assign({}, mb, { id: nextId++ });
      merged.push(copy);
    }
  }
  currentBlobs = merged;

  sampledBaseTone = {
    rgb: data.base_tone_rgb || [230, 180, 150],
    lab: data.base_tone_lab || [185.7, 141.0, 145.0]
  };
  updateToneSwatch(sampledBaseTone.rgb);

  currentSkinMaskImage = null;
  currentSkinMaskBlob = null;
  if (data.skin_mask_base64) {
    currentSkinMaskBlob = dataUrlToBlob(data.skin_mask_base64);
    currentSkinMaskImage = await loadImage(data.skin_mask_base64, "img-skin-mask-holder");
  }

  resultImage = null;
  renderCanvas();
  setLog(`Detected ${newAuto.length} spots (${data.skin_percentage}% skin) in ${data.process_time_ms}ms. Rendering result preview\u2026`, "success");

  // Immediately show the retouched look on the AFTER side
  logDebug("[ANALYZE] Launching Before/After result preview...", "info");
  await runPreview();
  logDebug("[ANALYZE] Ready for interactive editing!", "ok");
  return data;
}

btnAutoDetect.addEventListener("click", async () => {
  if (isProcessing) return;
  isProcessing = true;
  btnAutoDetect.disabled = true;
  detectSpinner.classList.add("active");
  detectBtnText.textContent = "Analyzing\u2026";
  setPreviewStatus("analyzing");

  try {
    await runAnalysis(true);
  } catch (err) {
    logDebug(`[ANALYZE ERROR] ${err.stack || err.message || err}`, "error");
    console.error("Analyze error:", err);
    setPreviewStatus("error");
    setLog(err.message, "error");
  } finally {
    isProcessing = false;
    btnAutoDetect.disabled = false;
    detectSpinner.classList.remove("active");
    detectBtnText.textContent = "1. Analyze Portrait";
  }
});

/* Sensitivity / neck changes re-run detection (preserving manual edits) */
function scheduleReanalyze(delay = 1200) {
  if (!currentPortraitBlob) return;
  clearTimeout(analyzeTimer);
  analyzeTimer = setTimeout(async () => {
    if (isProcessing) return;
    isProcessing = true;
    detectSpinner.classList.add("active");
    setPreviewStatus("analyzing");
    try {
      await runAnalysis(false);
    } catch (err) {
      logDebug(`[RE-ANALYZE ERROR] ${err.message || err}`, "error");
      setPreviewStatus("error");
      setLog(err.message, "error");
    } finally {
      isProcessing = false;
      detectSpinner.classList.remove("active");
    }
  }, delay);
}

/* ============================== CONTROL WIRING ============================== */
function setupAccordion(btnId, drawerId) {
  const btn = document.getElementById(btnId);
  const drawer = document.getElementById(drawerId);
  if (btn && drawer) {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      drawer.classList.toggle("open");
    });
  }
}

setupAccordion("btn-toggle-prompt", "drawer-prompt");
setupAccordion("btn-toggle-training", "drawer-training");
setupAccordion("btn-toggle-fine-tuning", "drawer-fine-tuning");
setupAccordion("btn-toggle-debug", "drawer-debug");

/* Diagnostics Runner */
async function runDiagnostics() {
  logDebug("=== STARTING FULL SYSTEM DIAGNOSTICS ===", "info");
  
  // 1. Check Photoshop Environment
  try {
    const psVer = (app && app.version) ? app.version : "Unknown";
    logDebug(`Photoshop Engine Version: ${psVer}`, "info");
    const activeDoc = app.activeDocument;
    if (activeDoc) {
      logDebug(`Active Document: "${activeDoc.name || 'Untitled'}" (${activeDoc.width}x${activeDoc.height}px, ${activeDoc.mode})`, "ok");
    } else {
      logDebug("No document currently open in Photoshop. Please open an image first.", "warn");
    }
  } catch (e) {
    logDebug(`Photoshop DOM check error: ${e.message}`, "error");
  }

  // 2. Check Backend Server Connection
  const serverUrl = getServerUrl();
  logDebug(`Testing server connection at: ${serverUrl}/health ...`, "info");
  try {
    const t0 = Date.now();
    const res = await fetch(`${serverUrl}/health`, { method: "GET" });
    const lat = Date.now() - t0;
    if (res.ok) {
      const info = await res.json();
      logDebug(`Server 200 OK (${lat}ms): Model=${info.model}, Device=${info.device}, Ready=${info.status}`, "ok");
    } else {
      logDebug(`Server returned error status ${res.status}: ${await res.text()}`, "error");
    }
  } catch (netErr) {
    logDebug(`Server connection FAILED (${serverUrl}): ${netErr.message || netErr}`, "error");
  }

  // 3. Test Snapshot Capture
  if (app.activeDocument) {
    logDebug("Testing document snapshot capture pipeline...", "info");
    try {
      const t0 = Date.now();
      const blob = await exportFullPortrait();
      const dt = Date.now() - t0;
      logDebug(`Snapshot test SUCCESS in ${dt}ms: Captured ${(blob.size / 1024).toFixed(1)} KB image.`, "ok");
    } catch (snapErr) {
      logDebug(`Snapshot test FAILED: ${snapErr.message || snapErr}`, "error");
    }
  }

  logDebug("=== DIAGNOSTICS COMPLETE ===", "info");
}

if (btnRunDiagnostics) {
  btnRunDiagnostics.addEventListener("click", runDiagnostics);
}

if (btnCopyDebugLogs) {
  btnCopyDebugLogs.addEventListener("click", async () => {
    try {
      const text = debugLogs.join("\n");
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
      }
      setLog("Debug logs copied to clipboard!", "success");
      logDebug("Debug logs copied to clipboard.", "ok");
    } catch (e) {
      logDebug(`Clipboard copy failed: ${e.message}`, "warn");
    }
  });
}

if (btnClearDebugLogs) {
  btnClearDebugLogs.addEventListener("click", () => {
    debugLogs.length = 0;
    if (debugTerminal) {
      debugTerminal.innerHTML = '<div class="debug-line info">[SYS] Logs cleared.</div>';
    }
  });
}

if (sliderTexture && valTexture) {
  sliderTexture.addEventListener("input", () => {
    valTexture.textContent = `${sliderTexture.value}%`;
    schedulePreview(650);
  });
}
if (sliderFeather && valFeather) {
  sliderFeather.addEventListener("input", () => {
    valFeather.textContent = `${sliderFeather.value}px`;
    schedulePreview(650);
  });
}

sliderGrain.addEventListener("input", () => {
  valGrain.textContent = `${sliderGrain.value}%`;
  schedulePreview(650);
});
sliderDbStrength.addEventListener("input", () => {
  valDbStrength.textContent = `${sliderDbStrength.value}%`;
  schedulePreview(650);
});
sliderStrength.addEventListener("input", () => {
  valStrength.textContent = `${sliderStrength.value}%`;
  schedulePreview(650);
});
sliderSmoothStrength.addEventListener("input", () => {
  valSmoothStrength.textContent = `${sliderSmoothStrength.value}%`;
  schedulePreview(650);
});
sliderTextureKeep.addEventListener("input", () => {
  valTextureKeep.textContent = `${sliderTextureKeep.value}%`;
  schedulePreview(650);
});
sliderTeeth.addEventListener("input", () => {
  valTeeth.textContent = `${sliderTeeth.value}%`;
  schedulePreview(650);
});
sliderEyes.addEventListener("input", () => {
  valEyes.textContent = `${sliderEyes.value}%`;
  schedulePreview(650);
});
sliderShine.addEventListener("input", () => {
  valShine.textContent = `${sliderShine.value}%`;
  schedulePreview(650);
});
sliderSensitivity.addEventListener("input", () => {
  valSensitivity.textContent = `${sliderSensitivity.value}%`;
  scheduleReanalyze(1400);
});

if (chkPreserveMoles) chkPreserveMoles.addEventListener("change", () => scheduleReanalyze(800));
chkIncludeNeck.addEventListener("change", () => scheduleReanalyze(900));
chkEnableHeal.addEventListener("change", () => schedulePreview(250));
chkEnableDb.addEventListener("change", () => schedulePreview(250));
chkEnableSmooth.addEventListener("change", () => schedulePreview(250));
chkEnableLighten.addEventListener("change", () => schedulePreview(250));
chkEnableEyesTeeth.addEventListener("change", () => schedulePreview(250));
chkEnableShine.addEventListener("change", () => schedulePreview(250));

/* ============================== LAYER 4: AI TEXT REFINEMENT ============================== */
btnSendPrompt.addEventListener("click", async () => {
  const promptText = inputPrompt.value.trim();
  if (!promptText) return;
  if (!currentPortraitBlob) {
    setLog("Click 'Analyze Portrait' first.", "warning");
    return;
  }

  setLog(`AI instruction: "${promptText}"\u2026`, "info");
  btnSendPrompt.disabled = true;

  try {
    const formData = new FormData();
    formData.append("image", currentPortraitBlob, "portrait.png");
    formData.append("prompt", promptText);
    formData.append("blobs_json", JSON.stringify(currentBlobs));

    const res = await fetch(`${getServerUrl()}/refine-text`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(`AI refinement failed: ${await res.text()}`);

    const data = await res.json();
    currentBlobs = data.blobs || currentBlobs;
    renderCanvas();
    setLog(`AI refinement applied (${data.action}). Preview updating\u2026`, "success");
    inputPrompt.value = "";
    schedulePreview(300);

  } catch (err) {
    console.error("Refine text error:", err);
    setLog(err.message, "error");
  } finally {
    btnSendPrompt.disabled = false;
  }
});

/* ============================== REVIEWED TRAINING EXPORT ============================== */
btnExportTraining.addEventListener("click", async () => {
  if (!currentPortraitBlob) {
    trainingStatus.textContent = "Analyze a portrait before exporting a training sample.";
    setLog("Analyze a portrait before saving training data.", "warning");
    return;
  }
  if (!chkTrainingConsent.checked) {
    trainingStatus.textContent = "Permission confirmation is required before export.";
    setLog("Confirm permission to use this portrait for training.", "warning");
    return;
  }
  if (!isServerOnline) {
    await checkServerStatus();
    if (!isServerOnline) {
      trainingStatus.textContent = "AI server is offline.";
      setLog("AI backend offline. Start backend\\run_server.bat first.", "error");
      return;
    }
  }

  btnExportTraining.disabled = true;
  trainingStatus.textContent = "Saving reviewed image and YOLO polygons\u2026";
  try {
    const formData = new FormData();
    formData.append("image", currentPortraitBlob, "reviewed_portrait.png");
    formData.append("blobs_json", JSON.stringify(currentBlobs));
    formData.append("split", selectTrainingSplit.value);
    formData.append("reviewed", "true");

    const res = await fetch(`${getServerUrl()}/training/export`, {
      method: "POST",
      body: formData
    });
    if (!res.ok) throw new Error(`Training export failed: ${await res.text()}`);

    const data = await res.json();
    trainingStatus.textContent = `Saved ${data.labels} reviewed labels to ${data.split}.`;
    setLog(`Training sample saved (${data.labels} labels).`, "success");
  } catch (err) {
    console.error("Training export error:", err);
    trainingStatus.textContent = err.message;
    setLog(err.message, "error");
  } finally {
    btnExportTraining.disabled = false;
  }
});

/* ============================== STEP 2: APPLY TO PHOTOSHOP ============================== */
btnApplyAll.addEventListener("click", async () => {
  if (isProcessing) return;

  if (!isServerOnline) {
    await checkServerStatus();
    if (!isServerOnline) {
      setLog("AI backend offline. Start it with backend\\run_server.bat", "error");
      return;
    }
  }
  const doHeal = chkEnableHeal.checked;
  const doDb = chkEnableDb.checked;
  const doSmooth = chkEnableSmooth.checked;
  const doLighten = chkEnableLighten.checked;
  const doEyesTeeth = chkEnableEyesTeeth.checked;
  const doShine = chkEnableShine.checked;

  if (!doHeal && !doDb && !doSmooth && !doLighten && !doEyesTeeth && !doShine) {
    setLog("Enable at least one feature first.", "warning");
    return;
  }

  isProcessing = true;
  btnApplyAll.disabled = true;
  applySpinner.classList.add("active");
  const startTime = Date.now();

  try {
    const doc = app.activeDocument;
    if (!doc) throw new Error("No active document open in Photoshop.");

    setLog("Capturing fresh full-resolution snapshot\u2026", "info");
    const portraitBlob = await exportFullPortrait();
    const placedNames = [];

    // Heal chain: when healing is enabled, downstream layers (D&B, Smooth,
    // Shine) must be computed from the HEALED base — otherwise the original
    // blemish pixels show back through their skin-mask alpha.
    const healChainBlobs = doHeal ? currentBlobs.filter(b => b.active !== false) : [];
    const healChainActive = healChainBlobs.length > 0;
    const healChainParams = {
      heal_mode: "full_inpaint",
      texture_blend: (parseInt(sliderTexture.value, 10) / 100).toString(),
      grain_intensity: (parseInt(sliderGrain.value, 10) / 100).toString()
    };

    /* ACTION 1: Remove Pimples (LaMa inpainting) */
    if (doHeal) {
      const activeBlobs = currentBlobs.filter(b => b.active !== false);
      if (activeBlobs.length > 0) {
        applyBtnText.textContent = `Healing ${activeBlobs.length} spots\u2026`;
        setLog(`AI inpainting ${activeBlobs.length} regions\u2026`, "info");

        const healForm = new FormData();
        healForm.append("image", portraitBlob, "portrait.png");
        healForm.append("blobs_json", JSON.stringify(activeBlobs));
        healForm.append("texture_blend", (parseInt(sliderTexture.value, 10) / 100).toString());
        healForm.append("feather_radius", sliderFeather.value);
        healForm.append("grain_intensity", (parseInt(sliderGrain.value, 10) / 100).toString());
        // Skin mask keeps the healer's baseline sampling on real skin pixels
        // (prevents grey patches next to nostrils / nose folds).
        if (currentSkinMaskBlob) healForm.append("skin_mask", currentSkinMaskBlob, "skin_mask.png");

        const healRes = await fetch(`${getServerUrl()}/apply-heal`, { method: "POST", body: healForm });
        if (!healRes.ok) throw new Error(`Healing failed: ${await healRes.text()}`);

        const healBuffer = await (await healRes.blob()).arrayBuffer();
        await placePatchAsLayer(healBuffer, "healed_layer.png", "AI Healed Blemishes");
        placedNames.push("AI Healed Blemishes");
      }
    }

    /* ACTION 2: AI Dodge & Burn (Micro-Contrast Tonal Evening) */
    if (doDb) {
      applyBtnText.textContent = "Applying Dodge & Burn\u2026";
      setLog("AI micro-contrast Dodge & Burn\u2026", "info");

      const dbForm = new FormData();
      dbForm.append("image", portraitBlob, "portrait.png");
      dbForm.append("strength", (parseInt(sliderDbStrength.value, 10) / 100).toString());
      dbForm.append("feather_radius", sliderFeather.value);
      if (currentSkinMaskBlob) dbForm.append("skin_mask", currentSkinMaskBlob, "skin_mask.png");
      if (healChainActive) {
        dbForm.append("blobs_json", JSON.stringify(healChainBlobs));
        dbForm.append("heal_mode", healChainParams.heal_mode);
        dbForm.append("texture_blend", healChainParams.texture_blend);
        dbForm.append("grain_intensity", healChainParams.grain_intensity);
      }

      const dbRes = await fetch(`${getServerUrl()}/apply-dodge-burn`, { method: "POST", body: dbForm });
      if (dbRes.ok) {
        const dbBuffer = await (await dbRes.blob()).arrayBuffer();
        await placePatchAsLayer(dbBuffer, "dodge_burn_layer.png", "AI Dodge & Burn");
        placedNames.push("AI Dodge & Burn");
      }
    }

    /* ACTION 3: Smooth Skin (frequency separation + redness even) */
    if (doSmooth) {
      applyBtnText.textContent = "Smoothing skin\u2026";
      setLog("Frequency-separation smoothing\u2026", "info");

      const smoothForm = new FormData();
      smoothForm.append("image", portraitBlob, "portrait.png");
      smoothForm.append("strength", (parseInt(sliderSmoothStrength.value, 10) / 100).toString());
      smoothForm.append("texture_keep", (parseInt(sliderTextureKeep.value, 10) / 100).toString());
      smoothForm.append("feather_radius", sliderFeather.value);
      if (currentSkinMaskBlob) smoothForm.append("skin_mask", currentSkinMaskBlob, "skin_mask.png");
      if (healChainActive) {
        smoothForm.append("blobs_json", JSON.stringify(healChainBlobs));
        smoothForm.append("heal_mode", healChainParams.heal_mode);
        smoothForm.append("texture_blend", healChainParams.texture_blend);
        smoothForm.append("grain_intensity", healChainParams.grain_intensity);
      }

      const smoothRes = await fetch(`${getServerUrl()}/apply-smooth`, { method: "POST", body: smoothForm });
      if (!smoothRes.ok) throw new Error(`Smoothing failed: ${await smoothRes.text()}`);

      const smoothBuffer = await (await smoothRes.blob()).arrayBuffer();
      await placePatchAsLayer(smoothBuffer, "smoothed_layer.png", "AI Smoothed Skin");
      placedNames.push("AI Smoothed Skin");
    }

    /* ACTION 4: Native Photoshop Curves adjustment, masked to detected skin. */
    if (doLighten) {
      applyBtnText.textContent = "Creating native Curves layer\u2026";
      setLog("Creating an editable, skin-masked Curves adjustment\u2026", "info");
      await createMaskedToneLiftLayer();
      placedNames.push("AI Skin Tone Lift");
    }

    /* ACTION 5: AI Eyes & Teeth Enhancer */
    if (doEyesTeeth) {
      applyBtnText.textContent = "Enhancing eyes & teeth\u2026";
      setLog("AI teeth whitening & iris sparkle\u2026", "info");

      const eyeForm = new FormData();
      eyeForm.append("image", portraitBlob, "portrait.png");
      eyeForm.append("teeth_whiten", (parseInt(sliderTeeth.value, 10) / 100).toString());
      eyeForm.append("eye_brighten", (parseInt(sliderEyes.value, 10) / 100).toString());
      eyeForm.append("feather_radius", "3");

      const eyeRes = await fetch(`${getServerUrl()}/apply-eye-teeth`, { method: "POST", body: eyeForm });
      if (eyeRes.ok) {
        const eyeBuffer = await (await eyeRes.blob()).arrayBuffer();
        await placePatchAsLayer(eyeBuffer, "eyes_teeth_layer.png", "AI Eyes & Teeth");
        placedNames.push("AI Eyes & Teeth");
      }
    }

    /* ACTION 6: AI Anti-Glare Shine Neutralizer */
    if (doShine) {
      applyBtnText.textContent = "Defusing shine\u2026";
      setLog("AI specular shine reduction\u2026", "info");

      const shineForm = new FormData();
      shineForm.append("image", portraitBlob, "portrait.png");
      shineForm.append("strength", (parseInt(sliderShine.value, 10) / 100).toString());
      shineForm.append("feather_radius", "4");
      if (currentSkinMaskBlob) shineForm.append("skin_mask", currentSkinMaskBlob, "skin_mask.png");
      if (healChainActive) {
        shineForm.append("blobs_json", JSON.stringify(healChainBlobs));
        shineForm.append("heal_mode", healChainParams.heal_mode);
        shineForm.append("texture_blend", healChainParams.texture_blend);
        shineForm.append("grain_intensity", healChainParams.grain_intensity);
      }

      const shineRes = await fetch(`${getServerUrl()}/apply-shine-neutralize`, { method: "POST", body: shineForm });
      if (shineRes.ok) {
        const shineBuffer = await (await shineRes.blob()).arrayBuffer();
        await placePatchAsLayer(shineBuffer, "anti_shine_layer.png", "AI Anti-Glare Shine");
        placedNames.push("AI Anti-Glare Shine");
      }
    }

    /* Group results under one tidy group */
    if (placedNames.length > 1) {
      applyBtnText.textContent = "Grouping layers\u2026";
      await groupRetouchLayers("AI Retouch", placedNames);
    }

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
    setLog(`Done in ${elapsed}s! Non-destructive layers (${placedNames.join(", ")}) created.`, "success");

  } catch (err) {
    console.error("Apply error:", err);
    setLog(err.message, "error");
  } finally {
    isProcessing = false;
    btnApplyAll.disabled = false;
    applySpinner.classList.remove("active");
    applyBtnText.textContent = "2. Apply to Photoshop";
  }
});

setPreviewStatus("idle");
