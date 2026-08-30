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

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
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
  const candidateUrls = [
    getServerUrl(),
    "http://127.0.0.1:8765",
    "http://127.0.0.1:8766",
    "http://127.0.0.1:8001"
  ].filter((v, i, a) => v && a.indexOf(v) === i);

  statusBadge.className = "status-badge checking";
  statusLabel.textContent = "Checking\u2026";

  for (const baseUrl of candidateUrls) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000);
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
  ctx.font = "700 9px Segoe UI, sans-serif";
  const padX = 6, padY = 4;
  const tw = ctx.measureText(text).width;
  const w = tw + padX * 2;
  const h = 14;
  const rx = alignRight ? x - w : x;
  ctx.fillStyle = "rgba(0,0,0,0.62)";
  ctx.beginPath();
  if (ctx.roundRect) { ctx.roundRect(rx, y, w, h, 3); } else { ctx.rect(rx, y, w, h); }
  ctx.fill();
  ctx.fillStyle = "#ffffff";
  ctx.textBaseline = "middle";
  ctx.fillText(text, rx + padX, y + h / 2 + 0.5);
}

function drawSplitDivider(ctx, splitX) {
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
  ctx.clearRect(0, 0, cw, ch);

  const imgW = currentPortraitImage.naturalWidth || currentPortraitImage.width;
  const imgH = currentPortraitImage.naturalHeight || currentPortraitImage.height;
  const scale = Math.min(cw / imgW, ch / imgH);
  const renderW = imgW * scale;
  const renderH = imgH * scale;
  const offX = (cw - renderW) / 2;
  const offY = (ch - renderH) / 2;
  viewGeom = { scale, offX, offY, renderW, renderH };

  // 1. BEFORE base
  ctx.drawImage(currentPortraitImage, offX, offY, renderW, renderH);

  // 2. AFTER region
  let overlayLimitX = cw;
  if (viewMode === "after") {
    if (resultImage) {
      ctx.drawImage(resultImage, offX, offY, renderW, renderH);
      overlayLimitX = -1;
    }
  } else if (viewMode === "split") {
    const splitX = splitCanvasX();
    if (resultImage) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(splitX, 0, cw - splitX, ch);
      ctx.clip();
      ctx.drawImage(resultImage, offX, offY, renderW, renderH);
      ctx.restore();
      overlayLimitX = splitX;
    }
  }

  // 3. Detection overlays (clipped to the BEFORE side)
  if (overlayLimitX > 0) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, overlayLimitX, ch);
    ctx.clip();

    if (chkShowSkin.checked && currentSkinMaskImage) {
      const offCanvas = document.createElement("canvas");
      offCanvas.width = cw;
      offCanvas.height = ch;
      const offCtx = offCanvas.getContext("2d");
      offCtx.drawImage(currentSkinMaskImage, offX, offY, renderW, renderH);
      offCtx.globalCompositeOperation = "source-in";
      offCtx.fillStyle = "rgba(6, 182, 212, 0.32)";
      offCtx.fillRect(0, 0, cw, ch);
      ctx.drawImage(offCanvas, 0, 0);
    }

    if (chkShowPimples.checked && currentBlobs.length > 0) {
      for (const blob of currentBlobs) {
        const bcx = blob.centroid[0] * scale + offX;
        const bcy = blob.centroid[1] * scale + offY;
        const br = Math.max(3, blob.radius * scale);
        if (bcx > overlayLimitX + br) continue;

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
          ctx.setLineDash([2, 2]);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    }
    ctx.restore();
  }

  // 4. Split UI chrome
  if (viewMode === "split") {
    const splitX = splitCanvasX();
    if (splitX > 52) drawChip(ctx, 8, 8, "BEFORE");
    if (cw - splitX > 58 && resultImage) drawChip(ctx, cw - 8, 8, "AFTER", true);
    drawSplitDivider(ctx, splitX);
  }

  const activeCount = currentBlobs.filter(b => b.active !== false).length;
  blobCounter.textContent = `${activeCount} Spots`;
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

  // Blemish tool: toggle nearest blob, or add a new one
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

  if (hitIndex !== -1) {
    currentBlobs[hitIndex].active = !currentBlobs[hitIndex].active;
    setLog(currentBlobs[hitIndex].active
      ? `Spot #${currentBlobs[hitIndex].id} will be removed.`
      : `Spot #${currentBlobs[hitIndex].id} ignored.`, "info");
  } else {
    const newId = currentBlobs.length > 0 ? Math.max(...currentBlobs.map(b => b.id)) + 1 : 1;
    const radius = Math.max(8, Math.round(imgW / 240));
    currentBlobs.push({
      id: newId,
      centroid: [origX, origY],
      bbox: [Math.max(0, origX - radius), Math.max(0, origY - radius), Math.min(imgW, origX + radius), Math.min(imgH, origY + radius)],
      radius: radius,
      confidence: 1.0,
      active: true,
      label: "pimple",
      source: "manual_click"
    });
    setLog(`Missed spot added at (${origX}, ${origY}).`, "success");
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
    const imgUrl = URL.createObjectURL(blob);
    const img = await loadImage(imgUrl);
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
  const origDocId = app.activeDocument ? app.activeDocument.id : null;
  const tempFolder = await localFileSystem.getTemporaryFolder();
  const portraitFile = await tempFolder.createFile("portrait_temp.png", { overwrite: true });
  const portraitToken = localFileSystem.createSessionToken(portraitFile);

  await core.executeAsModal(async () => {
    await action.batchPlay([
      {
        _obj: "duplicate",
        _target: [{ _ref: "document", _enum: "ordinal", _value: "targetEnum" }],
        name: "AI_Portrait_Export_Temp"
      }
    ], {});

    const dupDoc = app.activeDocument;
    dupDoc.layers.forEach(l => {
      if (l.name.startsWith("AI ")) l.visible = false;
    });

    try {
      await action.batchPlay([
        {
          _obj: "save",
          as: { _obj: "PNGFormat", method: { _enum: "PNGMethod", _value: "quick" } },
          in: { _path: portraitToken, _kind: "local" },
          saveStage: { _enum: "saveStageType", _value: "saveBegin" }
        }
      ], {});
    } finally {
      await action.batchPlay([
        { _obj: "close", saving: { _enum: "yesNo", _value: "no" } }
      ], {});
      if (origDocId !== null && app.activeDocument && app.activeDocument.id !== origDocId) {
        action.batchPlay([
          { _obj: "select", _target: [{ _ref: "document", _enum: "id", _value: origDocId }] }
        ], {}).catch(() => {});
      }
    }
  }, { commandName: "Export Portrait Snapshot" });

  const rawBytes = await portraitFile.read({ format: uxp.storage.formats.binary });
  return new Blob([rawBytes], { type: "image/png" });
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

function createSkinAlphaMaskBlob() {
  if (!currentSkinMaskImage) return null;

  const maskCanvas = document.createElement("canvas");
  const width = currentSkinMaskImage.naturalWidth || currentSkinMaskImage.width;
  const height = currentSkinMaskImage.naturalHeight || currentSkinMaskImage.height;
  maskCanvas.width = width;
  maskCanvas.height = height;
  const ctx = maskCanvas.getContext("2d");
  ctx.drawImage(currentSkinMaskImage, 0, 0, width, height);

  // Photoshop can load a placed layer's transparency as a selection. Convert
  // the grayscale segmentation mask into white pixels with mask alpha.
  const pixels = ctx.getImageData(0, 0, width, height);
  for (let i = 0; i < pixels.data.length; i += 4) {
    const alpha = pixels.data[i];
    pixels.data[i] = 255;
    pixels.data[i + 1] = 255;
    pixels.data[i + 2] = 255;
    pixels.data[i + 3] = alpha;
  }
  ctx.putImageData(pixels, 0, 0);
  return dataUrlToBlob(maskCanvas.toDataURL("image/png"));
}

async function loadSkinSelection(targetLayerName = null) {
  const maskBlob = createSkinAlphaMaskBlob();
  if (!maskBlob) throw new Error("Analyze Portrait first so Photoshop can build the skin mask.");

  const tempFolder = await localFileSystem.getTemporaryFolder();
  const maskFile = await tempFolder.createFile("ai_skin_selection.png", { overwrite: true });
  await maskFile.write(await maskBlob.arrayBuffer(), { format: uxp.storage.formats.binary });
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
  await core.executeAsModal(async () => {
    await action.batchPlay([
      {
        _obj: "select",
        _target: layerNames.map(n => ({ _ref: "layer", _enum: "name", _value: n })),
        makeVisible: false,
        _options: { dialogOptions: "dontDisplay" }
      },
      {
        _obj: "make",
        _target: [{ _ref: "layerSection" }],
        from: [{ _ref: "layer", _enum: "ordinal", _value: "targetEnum" }],
        name: groupName,
        _options: { dialogOptions: "dontDisplay" }
      }
    ], {});
  }, { commandName: `Group ${groupName}` });
}

/* ============================== STEP 1: ANALYZE (Layers 1 + 2) ============================== */
async function runAnalysis(isUserInitiated) {
  if (!isServerOnline) {
    await checkServerStatus();
    if (!isServerOnline) throw new Error("AI backend offline. Start it with backend\\run_server.bat");
  }
  const doc = app.activeDocument;
  if (!doc) throw new Error("No active document. Open a portrait photo in Photoshop first.");

  if (isUserInitiated) setLog("Capturing snapshot \u0026 running AI segmentation\u2026", "info");
  const portraitBlob = await exportFullPortrait();
  currentPortraitBlob = portraitBlob;

  const imgUrl = URL.createObjectURL(portraitBlob);
  currentPortraitImage = await loadImage(imgUrl);

  // Native-res offscreen copy for instant pixel sampling
  portraitDataCanvas = document.createElement("canvas");
  portraitDataCanvas.width = currentPortraitImage.naturalWidth;
  portraitDataCanvas.height = currentPortraitImage.naturalHeight;
  portraitDataCanvas.getContext("2d").drawImage(currentPortraitImage, 0, 0);

  resizeViewport(currentPortraitImage.naturalWidth, currentPortraitImage.naturalHeight);
  canvasEmptyOverlay.classList.add("hidden");
  renderCanvas();

  const formData = new FormData();
  formData.append("image", portraitBlob, "portrait.png");
  formData.append("sensitivity", (parseInt(sliderSensitivity.value, 10) / 100).toString());
  formData.append("detect_pimples", "true");
  formData.append("detect_skin", "true");
  formData.append("include_neck", chkIncludeNeck.checked ? "true" : "false");
  formData.append("preserve_moles", chkPreserveMoles && chkPreserveMoles.checked ? "true" : "false");
  formData.append("feather_radius", sliderFeather.value);

  const res = await fetch(`${getServerUrl()}/analyze`, { method: "POST", body: formData });
  if (!res.ok) throw new Error(`Analyze failed (${res.status}): ${await res.text()}`);
  const data = await res.json();

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
    currentSkinMaskImage = await loadImage(data.skin_mask_base64);
  }

  resultImage = null;
  renderCanvas();
  setLog(`Detected ${newAuto.length} spots (${data.skin_percentage}% skin) in ${data.process_time_ms}ms. Rendering result preview\u2026`, "success");

  // Immediately show the retouched look on the AFTER side
  await runPreview();
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
      setPreviewStatus("error");
      setLog(err.message, "error");
    } finally {
      isProcessing = false;
      detectSpinner.classList.remove("active");
    }
  }, delay);
}

/* ============================== CONTROL WIRING ============================== */
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

      const smoothRes = await fetch(`${getServerUrl()}/apply-smooth`, { method: "POST", body: smoothForm });
      if (!smoothRes.ok) throw new Error(`Smoothing failed: ${await smoothRes.text()}`);

      const smoothBuffer = await (await smoothRes.blob()).arrayBuffer();
      await placePatchAsLayer(smoothBuffer, "smoothed_layer.png", "AI Smoothed Skin");
      await attachSkinLayerMask("AI Smoothed Skin");
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

      const shineRes = await fetch(`${getServerUrl()}/apply-shine-neutralize`, { method: "POST", body: shineForm });
      if (shineRes.ok) {
        const shineBuffer = await (await shineRes.blob()).arrayBuffer();
        await placePatchAsLayer(shineBuffer, "anti_shine_layer.png", "AI Anti-Glare Shine");
        await attachSkinLayerMask("AI Anti-Glare Shine");
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
