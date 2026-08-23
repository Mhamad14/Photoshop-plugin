/**
 * AI Retouching Photoshop UXP Plugin - v2 Auto-Detection Architecture
 * Face & Skin Segmentation (BiSeNet), Hybrid Pimple Detection, Live Canvas Preview & LaMa Inpainting
 */

const photoshop = require("photoshop");
const { app, core, action } = photoshop;
const uxp = require("uxp");
const { localFileSystem } = uxp.storage;

// UI Elements
const statusBadge = document.getElementById("server-status");
const statusLabel = document.getElementById("status-label");
const inputServerUrl = document.getElementById("input-server-url");
const btnRefreshStatus = document.getElementById("btn-refresh-status");

const btnAutoDetect = document.getElementById("btn-auto-detect");
const detectSpinner = document.getElementById("detect-spinner");
const blobCounter = document.getElementById("blob-counter");
const toneSwatch = document.getElementById("tone-swatch");

const previewCanvas = document.getElementById("preview-canvas");
const canvasEmptyOverlay = document.getElementById("canvas-empty-overlay");
const toolPimple = document.getElementById("tool-pimple");
const toolEyedropper = document.getElementById("tool-eyedropper");
const chkShowPimples = document.getElementById("chk-show-pimples");
const chkShowSkin = document.getElementById("chk-show-skin");

const inputPrompt = document.getElementById("input-prompt");
const btnSendPrompt = document.getElementById("btn-send-prompt");

const chkEnableHeal = document.getElementById("chk-enable-heal");
const sliderSensitivity = document.getElementById("slider-sensitivity");
const valSensitivity = document.getElementById("val-sensitivity");
const sliderTexture = document.getElementById("slider-texture");
const valTexture = document.getElementById("val-texture");
const sliderFeather = document.getElementById("slider-feather");
const valFeather = document.getElementById("val-feather");
const sliderGrain = document.getElementById("slider-grain");
const valGrain = document.getElementById("val-grain");

const chkEnableLighten = document.getElementById("chk-enable-lighten");
const sliderStrength = document.getElementById("slider-strength");
const valStrength = document.getElementById("val-strength");
const chkIncludeNeck = document.getElementById("chk-include-neck");

const btnApplyAll = document.getElementById("btn-apply-all");
const applySpinner = document.getElementById("apply-spinner");
const logBox = document.getElementById("log-box");
const logMessage = document.getElementById("log-message");

// Plugin State
let isServerOnline = false;
let isProcessing = false;
let activeTool = "pimple"; // 'pimple' or 'eyedropper'

let currentPortraitImage = null;
let currentSkinMaskImage = null;
let currentBlobs = [];
let sampledBaseTone = { rgb: [230, 180, 150], lab: [180.0, 140.0, 145.0] };
let lastPortraitBlob = null;

// Sliders listeners
sliderSensitivity.addEventListener("input", () => { valSensitivity.textContent = `${sliderSensitivity.value}%`; });
sliderTexture.addEventListener("input", () => { valTexture.textContent = `${sliderTexture.value}%`; });
sliderFeather.addEventListener("input", () => { valFeather.textContent = `${sliderFeather.value}px`; });
sliderGrain.addEventListener("input", () => { valGrain.textContent = `${sliderGrain.value}%`; });
sliderStrength.addEventListener("input", () => { valStrength.textContent = `${sliderStrength.value}%`; });

chkShowPimples.addEventListener("change", renderCanvas);
chkShowSkin.addEventListener("change", renderCanvas);

// Tool toggles
toolPimple.addEventListener("click", () => {
  activeTool = "pimple";
  toolPimple.classList.add("active");
  toolEyedropper.classList.remove("active");
  previewCanvas.style.cursor = "crosshair";
});

toolEyedropper.addEventListener("click", () => {
  activeTool = "eyedropper";
  toolEyedropper.classList.add("active");
  toolPimple.classList.remove("active");
  previewCanvas.style.cursor = "cell";
});

function setLog(message, type = "info") {
  logMessage.textContent = message;
  logBox.className = `log-box ${type}`;
}

function getServerUrl() {
  return inputServerUrl.value.trim().replace(/\/+$/, "");
}

/**
 * Check backend server connectivity with smart port discovery
 */
async function checkServerStatus() {
  const currentUrl = getServerUrl();
  const candidateUrls = [
    currentUrl,
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8008",
    "http://127.0.0.1:8000"
  ];
  const uniqueUrls = [...new Set(candidateUrls)];

  statusBadge.className = "status-badge checking";
  statusLabel.textContent = "Checking...";

  for (const baseUrl of uniqueUrls) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000);

      const response = await fetch(`${baseUrl}/health`, {
        method: "GET",
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (response.ok) {
        const data = await response.json();
        isServerOnline = true;
        statusBadge.className = "status-badge online";
        statusLabel.textContent = `Online (${data.device || "AI"})`;
        statusBadge.title = `Model: ${data.model} | Device: ${data.device} | URL: ${baseUrl}`;
        if (inputServerUrl.value !== baseUrl) {
          inputServerUrl.value = baseUrl;
        }
        return;
      }
    } catch (e) {
      // Try next
    }
  }

  isServerOnline = false;
  statusBadge.className = "status-badge offline";
  statusLabel.textContent = "Offline";
  statusBadge.title = "Cannot connect to AI backend. Is server running?";
}

// Auto-check status
checkServerStatus();
setInterval(checkServerStatus, 12000);
btnRefreshStatus.addEventListener("click", checkServerStatus);
inputServerUrl.addEventListener("change", checkServerStatus);


/**
 * Layer 3: Interactive Live Preview HTML5 Canvas Rendering Engine
 */
function renderCanvas() {
  if (!currentPortraitImage) return;

  const canvas = previewCanvas;
  const ctx = canvas.getContext("2d");
  const cw = canvas.width;
  const ch = canvas.height;

  ctx.clearRect(0, 0, cw, ch);

  // Compute aspect ratio scaling
  const imgW = currentPortraitImage.naturalWidth || currentPortraitImage.width;
  const imgH = currentPortraitImage.naturalHeight || currentPortraitImage.height;
  const scale = Math.min(cw / imgW, ch / imgH);
  const renderW = imgW * scale;
  const renderH = imgH * scale;
  const offsetX = (cw - renderW) / 2;
  const offsetY = (ch - renderH) / 2;

  // 1. Draw Base Portrait
  ctx.drawImage(currentPortraitImage, offsetX, offsetY, renderW, renderH);

  // 2. Draw Skin Mask Overlay (Cyan tint)
  if (chkShowSkin.checked && currentSkinMaskImage) {
    ctx.save();
    // Create temporary offscreen canvas for skin coloring
    const offCanvas = document.createElement("canvas");
    offCanvas.width = cw;
    offCanvas.height = ch;
    const offCtx = offCanvas.getContext("2d");

    offCtx.drawImage(currentSkinMaskImage, offsetX, offsetY, renderW, renderH);
    offCtx.globalCompositeOperation = "source-in";
    offCtx.fillStyle = "rgba(6, 182, 212, 0.35)"; // Cyan overlay
    offCtx.fillRect(0, 0, cw, ch);

    ctx.drawImage(offCanvas, 0, 0);
    ctx.restore();
  }

  // 3. Draw Pimple Blobs Overlay
  let activeCount = 0;
  if (currentBlobs && currentBlobs.length > 0) {
    for (const blob of currentBlobs) {
      const cx = blob.centroid[0] * scale + offsetX;
      const cy = blob.centroid[1] * scale + offsetY;
      const r = Math.max(3, blob.radius * scale);

      if (blob.active !== false) {
        activeCount++;
        if (chkShowPimples.checked) {
          // Outer glow halo
          ctx.beginPath();
          ctx.arc(cx, cy, r + 2, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(244, 63, 94, 0.25)";
          ctx.fill();

          // Stroke circle
          ctx.beginPath();
          ctx.arc(cx, cy, r, 0, Math.PI * 2);
          ctx.strokeStyle = "#f43f5e";
          ctx.lineWidth = 1.8;
          ctx.stroke();

          // Center indicator
          ctx.beginPath();
          ctx.arc(cx, cy, 1.5, 0, Math.PI * 2);
          ctx.fillStyle = "#ffffff";
          ctx.fill();
        }
      } else {
        // Inactive / ignored blob
        if (chkShowPimples.checked) {
          ctx.beginPath();
          ctx.arc(cx, cy, r, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(161, 161, 170, 0.5)";
          ctx.lineWidth = 1.2;
          ctx.setLineDash([2, 2]);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    }
  }

  blobCounter.textContent = `${activeCount} Spots`;
}

/**
 * Layer 4 Refinement: Canvas Click Interaction
 */
previewCanvas.addEventListener("click", (e) => {
  if (!currentPortraitImage) return;

  const rect = previewCanvas.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  const imgW = currentPortraitImage.naturalWidth || currentPortraitImage.width;
  const imgH = currentPortraitImage.naturalHeight || currentPortraitImage.height;
  const scale = Math.min(previewCanvas.width / imgW, previewCanvas.height / imgH);
  const offsetX = (previewCanvas.width - imgW * scale) / 2;
  const offsetY = (previewCanvas.height - imgH * scale) / 2;

  // Convert click to original image pixel coordinates
  const origX = (clickX - offsetX) / scale;
  const origY = (clickY - offsetY) / scale;

  if (origX < 0 || origX >= imgW || origY < 0 || origY >= imgH) return;

  if (activeTool === "eyedropper") {
    // Sample Skin Tone at pixel
    const tempC = document.createElement("canvas");
    tempC.width = imgW;
    tempC.height = imgH;
    const tempCtx = tempC.getContext("2d");
    tempCtx.drawImage(currentPortraitImage, 0, 0);
    const pixel = tempCtx.getImageData(Math.floor(origX), Math.floor(origY), 1, 1).data;
    const rgb = [pixel[0], pixel[1], pixel[2]];

    sampledBaseTone.rgb = rgb;
    const hex = `#${((1 << 24) + (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]).toString(16).slice(1)}`;
    toneSwatch.style.backgroundColor = hex;
    setLog(`Sampled skin tone: ${hex} at (${Math.round(origX)}, ${Math.round(origY)})`, "success");

  } else {
    // Pimple Tool: Toggle nearest blob or add new
    let hitIndex = -1;
    let minDistance = Infinity;

    for (let i = 0; i < currentBlobs.length; i++) {
      const b = currentBlobs[i];
      const dist = Math.hypot(origX - b.centroid[0], origY - b.centroid[1]);
      const effectiveRadius = Math.max(10, b.radius + 6);
      if (dist <= effectiveRadius && dist < minDistance) {
        minDistance = dist;
        hitIndex = i;
      }
    }

    if (hitIndex !== -1) {
      // Toggle existing blob
      currentBlobs[hitIndex].active = !currentBlobs[hitIndex].active;
      const state = currentBlobs[hitIndex].active ? "enabled" : "ignored";
      setLog(`Blemish #${currentBlobs[hitIndex].id} ${state}.`, "info");
    } else {
      // Add new blob
      const newId = currentBlobs.length > 0 ? Math.max(...currentBlobs.map(b => b.id)) + 1 : 1;
      const newBlob = {
        id: newId,
        centroid: [Math.round(origX), Math.round(origY)],
        bbox: [Math.max(0, Math.round(origX - 8)), Math.max(0, Math.round(origY - 8)), Math.min(imgW, Math.round(origX + 8)), Math.min(imgH, Math.round(origY + 8))],
        radius: 7,
        confidence: 1.0,
        active: true,
        label: "manual",
        source: "click"
      };
      currentBlobs.push(newBlob);
      setLog(`Added new blemish #${newId} at (${Math.round(origX)}, ${Math.round(origY)}).`, "success");
    }

    renderCanvas();
  }
});


/**
 * Helper: Export full portrait snapshot from Photoshop
 */
async function exportFullPortrait() {
  const tempFolder = await localFileSystem.getTemporaryFolder();
  const portraitFile = await tempFolder.createFile("portrait_temp.png", { overwrite: true });
  const portraitToken = localFileSystem.createSessionToken(portraitFile);

  await core.executeAsModal(async () => {
    // Duplicate doc
    await action.batchPlay([
      {
        _obj: "duplicate",
        _target: [{ _ref: "document", _enum: "ordinal", _value: "targetEnum" }],
        name: "AI_Portrait_Export_Temp"
      }
    ], {});

    const dupDoc = app.activeDocument;
    // Hide any previous AI retouch layers to capture pure original
    dupDoc.layers.forEach(l => {
      if (l.name.startsWith("AI ") || l.name.includes("Mask")) {
        l.visible = false;
      }
    });

    // Save PNG quick export
    await action.batchPlay([
      {
        _obj: "save",
        as: {
          _obj: "PNGFormat",
          method: { _enum: "PNGMethod", _value: "quick" }
        },
        in: { _path: portraitToken, _kind: "local" },
        saveStage: { _enum: "saveStageType", _value: "saveBegin" }
      }
    ], {});

    // Close duplicate
    await action.batchPlay([
      {
        _obj: "close",
        saving: { _enum: "yesNo", _value: "no" }
      }
    ], {});
  }, { commandName: "Export Portrait Snapshot" });

  const rawBytes = await portraitFile.read({ format: uxp.storage.formats.binary });
  return new Blob([rawBytes], { type: "image/png" });
}


/**
 * 1. Auto-Detect Portrait Action
 */
btnAutoDetect.addEventListener("click", async () => {
  if (isProcessing) return;

  if (!isServerOnline) {
    await checkServerStatus();
    if (!isServerOnline) {
      setLog("AI backend is offline. Please run `run_server.bat`.", "error");
      return;
    }
  }

  isProcessing = true;
  btnAutoDetect.disabled = true;
  detectSpinner.classList.add("active");
  setLog("Exporting portrait snapshot and running Layer 1 & 2 AI segmentation...", "info");

  try {
    const doc = app.activeDocument;
    if (!doc) {
      throw new Error("No active document open. Please open a portrait photo.");
    }

    const portraitBlob = await exportFullPortrait();
    lastPortraitBlob = portraitBlob;

    // Load as Image element for canvas
    const imgUrl = URL.createObjectURL(portraitBlob);
    const imgElem = new Image();
    await new Promise((resolve, reject) => {
      imgElem.onload = resolve;
      imgElem.onerror = reject;
      imgElem.src = imgUrl;
    });
    currentPortraitImage = imgElem;

    // Adjust canvas resolution
    previewCanvas.width = 300;
    previewCanvas.height = Math.round(300 * (imgElem.naturalHeight / imgElem.naturalWidth));

    // Send to /analyze endpoint
    const formData = new FormData();
    formData.append("image", portraitBlob, "portrait.png");
    formData.append("sensitivity", (parseInt(sliderSensitivity.value, 10) / 100).toString());
    formData.append("detect_pimples", "true");
    formData.append("detect_skin", "true");
    formData.append("include_neck", chkIncludeNeck.checked ? "true" : "false");
    formData.append("feather_radius", sliderFeather.value);

    const res = await fetch(`${getServerUrl()}/analyze`, {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Analyze failed (${res.status}): ${errText}`);
    }

    const data = await res.json();
    currentBlobs = data.blobs || [];
    sampledBaseTone = {
      rgb: data.base_tone_rgb || [230, 180, 150],
      lab: data.base_tone_lab || [180.0, 140.0, 145.0]
    };

    // Update skin tone swatch
    const rgb = sampledBaseTone.rgb;
    const hex = `#${((1 << 24) + (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]).toString(16).slice(1)}`;
    toneSwatch.style.backgroundColor = hex;

    // Load skin mask image
    if (data.skin_mask_base64) {
      const skinImg = new Image();
      await new Promise((resolve) => {
        skinImg.onload = resolve;
        skinImg.src = data.skin_mask_base64;
      });
      currentSkinMaskImage = skinImg;
    }

    canvasEmptyOverlay.classList.add("hidden");
    renderCanvas();

    setLog(`Auto-detection complete in ${data.process_time_ms}ms: ${currentBlobs.length} blemish spots found, ${data.skin_percentage}% skin coverage.`, "success");

  } catch (err) {
    console.error("Auto-detect error:", err);
    setLog(err.message, "error");
  } finally {
    isProcessing = false;
    btnAutoDetect.disabled = false;
    detectSpinner.classList.remove("active");
  }
});


/**
 * Layer 4: AI Natural Language Refinement
 */
btnSendPrompt.addEventListener("click", async () => {
  const promptText = inputPrompt.value.trim();
  if (!promptText) return;
  if (!lastPortraitBlob) {
    setLog("Please click 'Auto-Detect Portrait' first.", "warning");
    return;
  }

  setLog(`Sending instruction to Gemini AI: "${promptText}"...`, "info");
  btnSendPrompt.disabled = true;

  try {
    const formData = new FormData();
    formData.append("image", lastPortraitBlob, "portrait.png");
    formData.append("prompt", promptText);
    formData.append("blobs_json", JSON.stringify(currentBlobs));

    const res = await fetch(`${getServerUrl()}/refine-text`, {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`AI Refinement error: ${errText}`);
    }

    const data = await res.json();
    currentBlobs = data.blobs || currentBlobs;
    renderCanvas();
    setLog(`AI refinement applied (${data.action}): Updated blemish selection.`, "success");
    inputPrompt.value = "";

  } catch (err) {
    console.error("Refine text error:", err);
    setLog(err.message, "error");
  } finally {
    btnSendPrompt.disabled = false;
  }
});


/**
 * Layer 5: Apply AI Retouch (Remove Pimples + Lighten Skin)
 */
btnApplyAll.addEventListener("click", async () => {
  if (isProcessing) return;

  if (!isServerOnline) {
    await checkServerStatus();
    if (!isServerOnline) {
      setLog("AI backend is offline. Please run `run_server.bat`.", "error");
      return;
    }
  }

  const doHeal = chkEnableHeal.checked;
  const doLighten = chkEnableLighten.checked;

  if (!doHeal && !doLighten) {
    setLog("Please enable at least one feature (Remove Pimples or Lighten Skin).", "warning");
    return;
  }

  isProcessing = true;
  btnApplyAll.disabled = true;
  applySpinner.classList.add("active");
  const startTime = Date.now();

  try {
    const doc = app.activeDocument;
    if (!doc) {
      throw new Error("No active document open in Photoshop.");
    }

    const tempFolder = await localFileSystem.getTemporaryFolder();

    // 1. Export fresh portrait snapshot
    setLog("Capturing portrait snapshot for AI application...", "info");
    const portraitBlob = await exportFullPortrait();

    // -------------------------------------------------------------
    // ACTION 1: Remove Pimples (LaMa Inpainting)
    // -------------------------------------------------------------
    if (doHeal) {
      const activeBlobs = currentBlobs.filter(b => b.active !== false);
      if (activeBlobs.length > 0) {
        setLog(`Executing AI Inpainting on ${activeBlobs.length} blemish regions...`, "info");

        const healForm = new FormData();
        healForm.append("image", portraitBlob, "portrait.png");
        healForm.append("blobs_json", JSON.stringify(activeBlobs));
        healForm.append("texture_blend", (parseInt(sliderTexture.value, 10) / 100).toString());
        healForm.append("feather_radius", sliderFeather.value);
        healForm.append("grain_intensity", (parseInt(sliderGrain.value, 10) / 100).toString());

        const healRes = await fetch(`${getServerUrl()}/apply-heal`, {
          method: "POST",
          body: healForm
        });

        if (!healRes.ok) {
          throw new Error(`Heal failed: ${await healRes.text()}`);
        }

        const healBlob = await healRes.blob();
        const healBuffer = await healBlob.arrayBuffer();
        const healFile = await tempFolder.createFile("healed_layer.png", { overwrite: true });
        await healFile.write(healBuffer, { format: uxp.storage.formats.binary });
        const healToken = localFileSystem.createSessionToken(healFile);

        // Place on new layer in Photoshop
        await core.executeAsModal(async () => {
          await action.batchPlay([
            {
              _obj: "placeEvent",
              target: { _path: healToken, _kind: "local" },
              _options: { dialogOptions: "dontDisplay" }
            }
          ], {});

          const placed = app.activeDocument.activeLayers[0];
          placed.name = "AI Healed Blemishes";
        }, { commandName: "Place AI Healed Layer" });

        setLog("Placed non-destructive layer 'AI Healed Blemishes'.", "info");
      }
    }

    // -------------------------------------------------------------
    // ACTION 2: Lighten Skin (Relative Tone Adjustment)
    // -------------------------------------------------------------
    if (doLighten) {
      setLog("Calculating relative skin lightening curves...", "info");

      const lightenForm = new FormData();
      lightenForm.append("image", portraitBlob, "portrait.png");
      lightenForm.append("strength", (parseInt(sliderStrength.value, 10) / 100).toString());
      lightenForm.append("base_tone_lab", JSON.stringify(sampledBaseTone.lab));
      lightenForm.append("feather_radius", sliderFeather.value);

      const lightenRes = await fetch(`${getServerUrl()}/apply-lighten`, {
        method: "POST",
        body: lightenForm
      });

      if (!lightenRes.ok) {
        throw new Error(`Lighten failed: ${await lightenRes.text()}`);
      }

      const lightenBlob = await lightenRes.blob();
      const lightenBuffer = await lightenBlob.arrayBuffer();
      const lightenFile = await tempFolder.createFile("lightened_layer.png", { overwrite: true });
      await lightenFile.write(lightenBuffer, { format: uxp.storage.formats.binary });
      const lightenToken = localFileSystem.createSessionToken(lightenFile);

      // Place on new layer in Photoshop
      await core.executeAsModal(async () => {
        await action.batchPlay([
          {
            _obj: "placeEvent",
            target: { _path: lightenToken, _kind: "local" },
            _options: { dialogOptions: "dontDisplay" }
          }
        ], {});

        const placed = app.activeDocument.activeLayers[0];
        placed.name = "AI Lightened Skin";
      }, { commandName: "Place AI Lightened Layer" });

      setLog("Placed non-destructive layer 'AI Lightened Skin'.", "info");
    }

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
    setLog(`✨ Retouch complete in ${elapsed}s! Non-destructive layers ready.`, "success");

  } catch (err) {
    console.error("Apply error:", err);
    setLog(err.message, "error");
  } finally {
    isProcessing = false;
    btnApplyAll.disabled = false;
    applySpinner.classList.remove("active");
  }
});
