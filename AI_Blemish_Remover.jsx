#target photoshop

/**
 * AI Retouch Pro v3.0 - Professional Studio Retouching Suite
 * Clean Dark Interface, Tabbed Architecture, Real-Time Progress Tracking.
 * Compatible with Photoshop CS6 through CC 2026.
 */

(function () {
    // Priority candidate URLs for local AI server
    var SERVER_CANDIDATES = [
        "http://127.0.0.1:8765",
        "http://127.0.0.1:8001",
        "http://127.0.0.1:8008",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:9001",
        "http://127.0.0.1:8080"
    ];
    var activeServerUrl = SERVER_CANDIDATES[0];

    var tempFolder = new Folder(Folder.temp.fsName + "/photoshop_ai_retouch");
    if (!tempFolder.exists) {
        tempFolder.create();
    }

    // -------------------------------------------------------------
    // Persistent Configuration Storage (Roaming User Data & Server)
    // -------------------------------------------------------------
    function loadStoredConfig() {
        var confDir = new Folder(Folder.userData.fsName + "/AI_Retouch_Pro");
        if (!confDir.exists) confDir.create();
        var confFile = new File(confDir.fsName + "/config.json");
        if (confFile.exists) {
            try {
                confFile.open("r");
                var content = confFile.read();
                confFile.close();
                if (content && content.length > 2) {
                    var parsed = eval("(" + content + ")");
                    if (parsed && typeof parsed === "object") return parsed;
                }
            } catch (e) {}
        }
        // Fallback: fetch from server config
        try {
            var getFile = new File(tempFolder.fsName + "/get_key.json");
            if (getFile.exists) getFile.remove();
            var getPath = getFile.fsName.replace(/\\/g, "/");
            app.system('curl.exe -s -m 2 "' + activeServerUrl + '/get-api-key" -o "' + getPath + '"');
            if (getFile.exists && getFile.length > 5) {
                getFile.open("r");
                var srvContent = getFile.read();
                getFile.close();
                try { getFile.remove(); } catch (e) {}
                var srvParsed = eval("(" + srvContent + ")");
                if (srvParsed && srvParsed.gemini_api_key) {
                    return { apiKey: srvParsed.gemini_api_key, lastPreset: 0, organizeGroup: true, multiLayer: true };
                }
            }
        } catch (e2) {}
        return { apiKey: "", lastPreset: 0, organizeGroup: true, multiLayer: true };
    }

    function saveStoredConfig(cfg) {
        var confDir = new Folder(Folder.userData.fsName + "/AI_Retouch_Pro");
        if (!confDir.exists) confDir.create();
        var confFile = new File(confDir.fsName + "/config.json");
        try {
            confFile.open("w");
            confFile.write('{"apiKey":"' + (cfg.apiKey || "") + '","lastPreset":' + (cfg.lastPreset || 0) + ',"organizeGroup":' + (cfg.organizeGroup ? "true" : "false") + ',"multiLayer":' + (cfg.multiLayer ? "true" : "false") + '}');
            confFile.close();
        } catch (e) {}
        if (cfg.apiKey) {
            saveApiKeyToServer(cfg.apiKey);
        }
    }

    // -------------------------------------------------------------
    // Health Check & Server Discovery
    // -------------------------------------------------------------
    function findActiveServer() {
        for (var i = 0; i < SERVER_CANDIDATES.length; i++) {
            var url = SERVER_CANDIDATES[i];
            var testFile = new File(tempFolder.fsName + "/health_" + i + ".json");
            if (testFile.exists) testFile.remove();

            var testPath = testFile.fsName.replace(/\\/g, "/");
            var cmd = 'curl.exe -s -m 2 "' + url + '/health" -o "' + testPath + '"';
            app.system(cmd);

            if (testFile.exists && testFile.length > 5) {
                testFile.open("r");
                var content = testFile.read();
                testFile.close();
                try { testFile.remove(); } catch (e) {}
                if (content.indexOf("ready") !== -1 || content.indexOf("status") !== -1) {
                    activeServerUrl = url;
                    var hasGemini = (content.indexOf('"gemini_enabled": true') !== -1 || content.indexOf('"gemini_enabled":true') !== -1);
                    var isCuda = (content.indexOf('"cuda_available": true') !== -1 || content.indexOf('"cuda_available":true') !== -1);
                    return { online: true, url: url, gemini: hasGemini, cuda: isCuda, info: content };
                }
            }
        }
        return { online: false, url: activeServerUrl, gemini: false, cuda: false, info: "Server offline" };
    }

    function saveApiKeyToServer(key) {
        var respFile = new File(tempFolder.fsName + "/set_key_resp.json");
        if (respFile.exists) respFile.remove();
        var respPath = respFile.fsName.replace(/\\/g, "/");

        var cmd = 'curl.exe -s -X POST -F "gemini_api_key=' + key + '" -o "' + respPath + '" "' + activeServerUrl + '/set-api-key"';
        app.system(cmd);

        if (respFile.exists) {
            respFile.remove();
            return true;
        }
        return false;
    }

    function importLayerFromFile(file, layerName, opacityVal, targetGroup, blendMode) {
        if (!file || !file.exists || file.length < 300) {
            return null;
        }
        var doc = app.activeDocument;
        var origRuler = app.preferences.rulerUnits;
        try { app.preferences.rulerUnits = Units.PIXELS; } catch (e) {}

        var tempDoc = app.open(file);
        var importedLayer = null;

        try {
            // Convert Background layer to normal ArtLayer so Photoshop allows duplication
            if (tempDoc.activeLayer.isBackgroundLayer) {
                tempDoc.activeLayer.isBackgroundLayer = false;
            }
            importedLayer = tempDoc.activeLayer.duplicate(doc, ElementPlacement.PLACEATBEGINNING);
            tempDoc.close(SaveOptions.DONOTSAVECHANGES);
        } catch (dupErr) {
            // Direct clipboard fallback if duplicate is blocked
            try {
                tempDoc.selection.selectAll();
                tempDoc.selection.copy();
                tempDoc.close(SaveOptions.DONOTSAVECHANGES);
                doc.paste();
                importedLayer = doc.activeLayer;
            } catch (copyErr) {
                try { tempDoc.close(SaveOptions.DONOTSAVECHANGES); } catch (e3) {}
                alert("Layer import failed: " + dupErr.message, "AI Import Error");
                return null;
            }
        }

        try { app.preferences.rulerUnits = origRuler; } catch (e) {}

        if (!importedLayer) return null;

        doc.activeLayer = importedLayer;
        if (layerName) importedLayer.name = layerName;
        if (opacityVal !== undefined && opacityVal !== null) importedLayer.opacity = opacityVal;
        if (blendMode) {
            try { importedLayer.blendMode = blendMode; } catch (e) {}
        }
        if (targetGroup) {
            try { importedLayer.move(targetGroup, ElementPlacement.INSIDE); } catch (e) {}
        }
        return importedLayer;
    }

    function placeFile(file) {
        // Robust fallback calling importLayerFromFile
        return importLayerFromFile(file, null, 100, null, null);
    }

    function validateActiveDocument() {
        if (app.documents.length === 0) {
            alert("Please open a portrait photograph in Photoshop first.", "AI Retouch Pro");
            return false;
        }
        var doc = app.activeDocument;
        if (doc.mode !== DocumentMode.RGB) {
            var convert = confirm("The active document is not in RGB mode (" + doc.mode + ").\nConvert to RGB Color to apply AI Retouching?");
            if (convert) {
                doc.changeMode(ChangeMode.RGB);
            } else {
                return false;
            }
        }
        return true;
    }

    function exportCleanPortrait(targetFile, forceFresh) {
        if (!forceFresh && targetFile.exists && targetFile.length > 500) {
            return true; // Fast cache hit
        }
        if (targetFile.exists) targetFile.remove();

        var doc = app.activeDocument;
        var pngSaveOptions = new PNGSaveOptions();
        pngSaveOptions.compression = 0;
        pngSaveOptions.interlaced = false;

        var dupDoc = doc.duplicate("AI_Temp_Export", false);
        try { dupDoc.selection.deselect(); } catch (e) {}
        try {
            if (dupDoc.bitsPerChannel !== BitsPerChannelType.EIGHT) {
                dupDoc.bitsPerChannel = BitsPerChannelType.EIGHT;
            }
        } catch (e) {}

        for (var k = 0; k < dupDoc.artLayers.length; k++) {
            var l = dupDoc.artLayers[k];
            if (!l.isBackgroundLayer && (l.name.indexOf("AI ") === 0 || l.name.indexOf("Mask") !== -1)) {
                l.visible = false;
            }
        }
        dupDoc.flatten();
        dupDoc.saveAs(targetFile, pngSaveOptions, true, Extension.LOWERCASE);
        dupDoc.close(SaveOptions.DONOTSAVECHANGES);
        return targetFile.exists && targetFile.length > 500;
    }

    function getOrCreateRetouchGroup() {
        var doc = app.activeDocument;
        var groupName = "AI Retouch Pro";
        var group = null;
        for (var i = 0; i < doc.layerSets.length; i++) {
            if (doc.layerSets[i].name === groupName) {
                group = doc.layerSets[i];
                break;
            }
        }
        if (!group) {
            group = doc.layerSets.add();
            group.name = groupName;
        }
        return group;
    }

    function getCurlErrorLog() {
        var errFile = new File(tempFolder.fsName + "/curl_err.log");
        if (errFile.exists) {
            errFile.open("r");
            var txt = errFile.read();
            errFile.close();
            return txt;
        }
        return "";
    }

    // -------------------------------------------------------------
    // STUDIO ACTIONS
    // -------------------------------------------------------------

    var curlBinary = ($.os && $.os.indexOf("Windows") !== -1) ? "curl.exe" : "curl";

    // Unified Master Pipeline
    function executeMasterSuite(params, targetGroup) {
        if (!validateActiveDocument()) return false;

        var fullExportFile = new File(tempFolder.fsName + "/full_portrait.png");
        var masterResultFile = new File(tempFolder.fsName + "/master_retouch.png");
        var errLogFile = new File(tempFolder.fsName + "/curl_err.log");

        if (masterResultFile.exists) masterResultFile.remove();
        if (errLogFile.exists) errLogFile.remove();

        if (!exportCleanPortrait(fullExportFile, true)) {
            alert("Failed to export portrait snapshot from Photoshop.", "Export Error");
            return false;
        }

        var exportPath = fullExportFile.fsName.replace(/\\/g, "/");
        var outputPath = masterResultFile.fsName.replace(/\\/g, "/");
        var errPath = errLogFile.fsName.replace(/\\/g, "/");

        var cmd = curlBinary + ' -s -S --tcp-nodelay -m 120 -X POST ' +
            '-F "image=@' + exportPath + '" ' +
            '-F "include_heal=' + (params.includeHeal ? "true" : "false") + '" ' +
            '-F "sensitivity=' + params.sensitivity + '" ' +
            '-F "heal_mode=' + params.healMode + '" ' +
            '-F "texture_blend=' + params.textureBlend + '" ' +
            '-F "include_dodge_burn=' + (params.includeDb ? "true" : "false") + '" ' +
            '-F "db_strength=' + params.dbStrength + '" ' +
            '-F "include_smooth=' + (params.includeSmooth ? "true" : "false") + '" ' +
            '-F "smooth_strength=' + params.smoothStrength + '" ' +
            '-F "texture_keep=' + params.textureKeep + '" ' +
            '-F "include_lighten=' + (params.includeLighten ? "true" : "false") + '" ' +
            '-F "lighten_strength=' + params.lightenStrength + '" ' +
            '-F "include_eye_teeth=' + (params.includeEyeTeeth ? "true" : "false") + '" ' +
            '-F "teeth_whiten=' + params.etStrength + '" ' +
            '-F "eye_brighten=' + params.etStrength + '" ' +
            '-F "include_shine=' + (params.includeShine ? "true" : "false") + '" ' +
            '-F "shine_strength=' + params.shineStrength + '" ' +
            '-F "gemini_api_key=' + params.apiKey + '" ' +
            '-o "' + outputPath + '" ' +
            '"' + activeServerUrl + '/apply-complete-suite" 2> "' + errPath + '"';

        app.system(cmd);

        if (!masterResultFile.exists || masterResultFile.length < 500) {
            var errLog = getCurlErrorLog();
            alert("Master Retouch Suite failed.\nServer: " + activeServerUrl + "\n\nDetails: " + (errLog || "Server unreachable. Start backend\\run_server.bat"), "AI Server Notice");
            return false;
        }

        var imported = importLayerFromFile(masterResultFile, "AI Master Retouch Composite", 100, targetGroup, BlendMode.NORMAL);
        return imported !== null;
    }

    // 1. AUTO-HEAL: AI Blemish & Acne Inpainting
    function executeAutoHeal(sensitivity, textureBlend, featherRadius, opacityVal, apiKey, targetGroup, healMode) {
        healMode = healMode || "full_inpaint";
        if (!validateActiveDocument()) return false;

        var fullExportFile = new File(tempFolder.fsName + "/full_portrait.png");
        var autoHealedFile = new File(tempFolder.fsName + "/auto_healed.png");
        var errLogFile = new File(tempFolder.fsName + "/curl_err.log");

        if (autoHealedFile.exists) autoHealedFile.remove();
        if (errLogFile.exists) errLogFile.remove();

        exportCleanPortrait(fullExportFile, false);

        var exportPath = fullExportFile.fsName.replace(/\\/g, "/");
        var outputPath = autoHealedFile.fsName.replace(/\\/g, "/");
        var errPath = errLogFile.fsName.replace(/\\/g, "/");

        var cmd = curlBinary + ' -s -S --tcp-nodelay -m 120 -X POST ' +
            '-F "image=@' + exportPath + '" ' +
            '-F "sensitivity=' + sensitivity + '" ' +
            '-F "heal_mode=' + healMode + '" ' +
            '-F "texture_blend=' + textureBlend + '" ' +
            '-F "feather_radius=' + featherRadius + '" ' +
            '-F "gemini_api_key=' + apiKey + '" ' +
            '-F "grain_intensity=0.03" ' +
            '-o "' + outputPath + '" ' +
            '"' + activeServerUrl + '/auto-heal" 2> "' + errPath + '"';

        app.system(cmd);

        if (!autoHealedFile.exists || autoHealedFile.length < 500) {
            var errLog = getCurlErrorLog();
            alert("Auto-heal failed.\nServer: " + activeServerUrl + "\n\nDetails: " + (errLog || "Verify AI server is active."), "AI Server Notice");
            return false;
        }

        var layerTitle = healMode === "calm_redness" ? "1. AI Calmed Redness" : (healMode === "flatten_bump" ? "1. AI Flattened Blemishes" : "1. AI Blemish Inpaint");
        var blend = healMode === "calm_redness" ? BlendMode.COLOR : BlendMode.NORMAL;
        var imported = importLayerFromFile(autoHealedFile, layerTitle, opacityVal, targetGroup, blend);
        return imported !== null;
    }

    // 2. SMOOTH SKIN: Frequency Separation
    function executeSmoothSkin(strength, textureKeep, featherRadius, opacityVal, targetGroup) {
        if (!validateActiveDocument()) return false;

        var fullExportFile = new File(tempFolder.fsName + "/full_portrait.png");
        var smoothedFile = new File(tempFolder.fsName + "/smoothed_skin.png");
        var errLogFile = new File(tempFolder.fsName + "/curl_err.log");

        if (smoothedFile.exists) smoothedFile.remove();
        if (errLogFile.exists) errLogFile.remove();

        exportCleanPortrait(fullExportFile, false);

        var exportPath = fullExportFile.fsName.replace(/\\/g, "/");
        var outputPath = smoothedFile.fsName.replace(/\\/g, "/");
        var errPath = errLogFile.fsName.replace(/\\/g, "/");

        var cmd = curlBinary + ' -s -S --tcp-nodelay -m 120 -X POST ' +
            '-F "image=@' + exportPath + '" ' +
            '-F "strength=' + strength + '" ' +
            '-F "texture_keep=' + textureKeep + '" ' +
            '-F "feather_radius=' + featherRadius + '" ' +
            '-o "' + outputPath + '" ' +
            '"' + activeServerUrl + '/apply-smooth" 2> "' + errPath + '"';

        app.system(cmd);

        if (!smoothedFile.exists || smoothedFile.length < 500) {
            var errLog = getCurlErrorLog();
            alert("Skin smoothing failed.\nServer: " + activeServerUrl + "\n\nDetails: " + (errLog || "Verify AI server is active."), "AI Server Notice");
            return false;
        }

        var imported = importLayerFromFile(smoothedFile, "2. AI Smooth Skin (Texture-Preserved)", opacityVal, targetGroup, BlendMode.NORMAL);
        return imported !== null;
    }

    // 3. LIGHTEN SKIN: Tone Lift
    function executeLightenSkin(strength, featherRadius, opacityVal, targetGroup) {
        if (!validateActiveDocument()) return false;

        var fullExportFile = new File(tempFolder.fsName + "/full_portrait.png");
        var lightenedFile = new File(tempFolder.fsName + "/lightened_skin.png");
        var errLogFile = new File(tempFolder.fsName + "/curl_err.log");

        if (lightenedFile.exists) lightenedFile.remove();
        if (errLogFile.exists) errLogFile.remove();

        exportCleanPortrait(fullExportFile, false);

        var exportPath = fullExportFile.fsName.replace(/\\/g, "/");
        var outputPath = lightenedFile.fsName.replace(/\\/g, "/");
        var errPath = errLogFile.fsName.replace(/\\/g, "/");

        var cmd = curlBinary + ' -s -S --tcp-nodelay -m 120 -X POST ' +
            '-F "image=@' + exportPath + '" ' +
            '-F "strength=' + strength + '" ' +
            '-F "feather_radius=' + featherRadius + '" ' +
            '-o "' + outputPath + '" ' +
            '"' + activeServerUrl + '/apply-lighten" 2> "' + errPath + '"';

        app.system(cmd);

        if (!lightenedFile.exists || lightenedFile.length < 500) {
            var errLog = getCurlErrorLog();
            alert("Skin lightening failed.\nServer: " + activeServerUrl + "\n\nDetails: " + (errLog || "Verify AI server is active."), "AI Server Notice");
            return false;
        }

        var imported = importLayerFromFile(lightenedFile, "3. AI Tone Lighten", opacityVal, targetGroup, BlendMode.SOFTLIGHT);
        return imported !== null;
    }

    // 4. DODGE & BURN: Micro-Contrast Sculpting
    function executeDodgeAndBurn(strength, softness, featherRadius, opacityVal, targetGroup) {
        if (!validateActiveDocument()) return false;

        var fullExportFile = new File(tempFolder.fsName + "/full_portrait.png");
        var dbFile = new File(tempFolder.fsName + "/dodge_burn.png");
        var errLogFile = new File(tempFolder.fsName + "/curl_err.log");

        if (dbFile.exists) dbFile.remove();
        if (errLogFile.exists) errLogFile.remove();

        exportCleanPortrait(fullExportFile, false);

        var exportPath = fullExportFile.fsName.replace(/\\/g, "/");
        var outputPath = dbFile.fsName.replace(/\\/g, "/");
        var errPath = errLogFile.fsName.replace(/\\/g, "/");

        var cmd = curlBinary + ' -s -S --tcp-nodelay -m 120 -X POST ' +
            '-F "image=@' + exportPath + '" ' +
            '-F "strength=' + strength + '" ' +
            '-F "softness=' + softness + '" ' +
            '-F "feather_radius=' + featherRadius + '" ' +
            '-o "' + outputPath + '" ' +
            '"' + activeServerUrl + '/apply-dodge-burn" 2> "' + errPath + '"';

        app.system(cmd);

        if (!dbFile.exists || dbFile.length < 500) {
            var errLog = getCurlErrorLog();
            alert("Dodge & Burn failed.\nServer: " + activeServerUrl + "\n\nDetails: " + (errLog || "Verify AI server is active."), "AI Server Notice");
            return false;
        }

        var imported = importLayerFromFile(dbFile, "4. AI Dodge & Burn", opacityVal, targetGroup, BlendMode.SOFTLIGHT);
        return imported !== null;
    }

    // 5. EYE & TEETH ENHANCEMENT
    function executeEyeAndTeeth(teethWhiten, eyeBrighten, irisSparkle, featherRadius, opacityVal, targetGroup) {
        if (!validateActiveDocument()) return false;

        var fullExportFile = new File(tempFolder.fsName + "/full_portrait.png");
        var etFile = new File(tempFolder.fsName + "/eye_teeth.png");
        var errLogFile = new File(tempFolder.fsName + "/curl_err.log");

        if (etFile.exists) etFile.remove();
        if (errLogFile.exists) errLogFile.remove();

        exportCleanPortrait(fullExportFile, false);

        var exportPath = fullExportFile.fsName.replace(/\\/g, "/");
        var outputPath = etFile.fsName.replace(/\\/g, "/");
        var errPath = errLogFile.fsName.replace(/\\/g, "/");

        var cmd = curlBinary + ' -s -S --tcp-nodelay -m 120 -X POST ' +
            '-F "image=@' + exportPath + '" ' +
            '-F "teeth_whiten=' + teethWhiten + '" ' +
            '-F "eye_brighten=' + eyeBrighten + '" ' +
            '-F "iris_sparkle=' + irisSparkle + '" ' +
            '-F "feather_radius=' + featherRadius + '" ' +
            '-o "' + outputPath + '" ' +
            '"' + activeServerUrl + '/apply-eye-teeth" 2> "' + errPath + '"';

        app.system(cmd);

        if (!etFile.exists || etFile.length < 500) {
            var errLog = getCurlErrorLog();
            alert("Eye & Teeth enhancement failed.\nServer: " + activeServerUrl + "\n\nDetails: " + (errLog || "Verify AI server is active."), "AI Server Notice");
            return false;
        }

        var imported = importLayerFromFile(etFile, "5. AI Eyes & Teeth Whitening", opacityVal, targetGroup, BlendMode.NORMAL);
        return imported !== null;
    }

    // 6. SHINE NEUTRALIZER
    function executeShineNeutralize(strength, threshold, featherRadius, opacityVal, targetGroup) {
        if (!validateActiveDocument()) return false;

        var fullExportFile = new File(tempFolder.fsName + "/full_portrait.png");
        var shineFile = new File(tempFolder.fsName + "/shine_neutral.png");
        var errLogFile = new File(tempFolder.fsName + "/curl_err.log");

        if (shineFile.exists) shineFile.remove();
        if (errLogFile.exists) errLogFile.remove();

        exportCleanPortrait(fullExportFile, false);

        var exportPath = fullExportFile.fsName.replace(/\\/g, "/");
        var outputPath = shineFile.fsName.replace(/\\/g, "/");
        var errPath = errLogFile.fsName.replace(/\\/g, "/");

        var cmd = curlBinary + ' -s -S --tcp-nodelay -m 120 -X POST ' +
            '-F "image=@' + exportPath + '" ' +
            '-F "strength=' + strength + '" ' +
            '-F "threshold=' + threshold + '" ' +
            '-F "feather_radius=' + featherRadius + '" ' +
            '-o "' + outputPath + '" ' +
            '"' + activeServerUrl + '/apply-shine-neutralize" 2> "' + errPath + '"';

        app.system(cmd);

        if (!shineFile.exists || shineFile.length < 500) {
            var errLog = getCurlErrorLog();
            alert("Shine neutralization failed.\nServer: " + activeServerUrl + "\n\nDetails: " + (errLog || "Verify AI server is active."), "AI Server Notice");
            return false;
        }

        var imported = importLayerFromFile(shineFile, "6. AI Shine Neutralizer", opacityVal, targetGroup, BlendMode.DARKEN);
        return imported !== null;
    }

    // 7. PREVIEW DETECTED MASKS
    function previewDetectedMask(sensitivity, apiKey, targetGroup) {
        if (!validateActiveDocument()) return false;

        var fullExportFile = new File(tempFolder.fsName + "/full_portrait.png");
        var maskPreviewFile = new File(tempFolder.fsName + "/detected_mask.png");
        var errLogFile = new File(tempFolder.fsName + "/curl_err.log");

        if (maskPreviewFile.exists) maskPreviewFile.remove();
        if (errLogFile.exists) errLogFile.remove();

        exportCleanPortrait(fullExportFile, false);

        var exportPath = fullExportFile.fsName.replace(/\\/g, "/");
        var outputPath = maskPreviewFile.fsName.replace(/\\/g, "/");
        var errPath = errLogFile.fsName.replace(/\\/g, "/");

        var cmd = curlBinary + ' -s -S --tcp-nodelay -m 120 -X POST ' +
            '-F "image=@' + exportPath + '" ' +
            '-F "sensitivity=' + sensitivity + '" ' +
            '-F "gemini_api_key=' + apiKey + '" ' +
            '-o "' + outputPath + '" ' +
            '"' + activeServerUrl + '/detect-mask" 2> "' + errPath + '"';

        app.system(cmd);

        if (maskPreviewFile.exists && maskPreviewFile.length > 100) {
            var imported = importLayerFromFile(maskPreviewFile, "AI Blemish Detection Mask (Preview)", 80, targetGroup, BlendMode.NORMAL);
            return imported !== null;
        }
        var errLog = getCurlErrorLog();
        alert("Blemish mask preview failed.\nServer: " + activeServerUrl + "\n\nDetails: " + (errLog || "Verify server is running."), "AI Server Notice");
        return false;
    }

    // -------------------------------------------------------------
    // DIALOG WINDOW CONSTRUCTION (Native TabbedPanel & 2-Column Grid)
    // -------------------------------------------------------------
    var dlg = new Window("dialog", "AI Retouch Pro - Studio Suite");
    dlg.orientation = "column";
    dlg.alignChildren = ["fill", "top"];
    dlg.spacing = 10;
    dlg.margins = 14;

    var cBg = [0.12, 0.13, 0.15, 1.0];
    var cPanel = [0.16, 0.17, 0.20, 1.0];
    var cTextPrimary = [0.96, 0.97, 0.98, 1.0];
    var cTextMuted = [0.65, 0.68, 0.72, 1.0];
    var cGreen = [0.25, 0.80, 0.45, 1.0];
    var cRed = [0.90, 0.35, 0.35, 1.0];

    try {
        dlg.graphics.backgroundColor = dlg.graphics.newBrush(dlg.graphics.BrushType.SOLID_COLOR, cBg);
    } catch (e) {}

    // Header & Status Bar
    var pnlHeader = dlg.add("group");
    pnlHeader.orientation = "row";
    pnlHeader.alignChildren = ["fill", "center"];
    pnlHeader.spacing = 10;

    var lblTitle = pnlHeader.add("statictext", undefined, "AI RETOUCH PRO  |  STUDIO SUITE");
    try {
        lblTitle.graphics.font = ScriptUI.newFont("Segoe UI", "BOLD", 13);
        lblTitle.graphics.foregroundColor = dlg.graphics.newPen(dlg.graphics.PenType.SOLID_COLOR, cTextPrimary, 1);
    } catch (e) {}

    var health = findActiveServer();
    var statusStr = health.online ? "[ONLINE: " + health.url.replace("http://127.0.0.1:", "PORT ") + (health.cuda ? " - CUDA GPU" : " - CPU") + "]" : "[OFFLINE: START SERVER]";
    var lblStatusBadge = pnlHeader.add("statictext", undefined, statusStr);
    lblStatusBadge.alignment = ["right", "center"];
    try {
        lblStatusBadge.graphics.font = ScriptUI.newFont("Segoe UI", "BOLD", 10);
        lblStatusBadge.graphics.foregroundColor = dlg.graphics.newPen(dlg.graphics.PenType.SOLID_COLOR, health.online ? cGreen : cRed, 1);
    } catch (e) {}

    // Native Tabbed Panel (Responsive, No OS scaling clipping)
    var tabFolder = dlg.add("tabbedpanel");
    tabFolder.alignChildren = ["fill", "fill"];
    tabFolder.preferredSize = [460, 260];

    // =============================================================
    // TAB 1: STUDIO PRESETS & ACTIONS
    // =============================================================
    var viewTab1 = tabFolder.add("tab", undefined, "Studio Presets");
    viewTab1.orientation = "column";
    viewTab1.alignChildren = ["fill", "top"];
    viewTab1.spacing = 8;
    viewTab1.margins = 10;

    var grpPresetRow = viewTab1.add("group");
    grpPresetRow.orientation = "row";
    grpPresetRow.alignChildren = ["left", "center"];
    var lblPreset = grpPresetRow.add("statictext", undefined, "Retouch Preset:");
    lblPreset.preferredSize.width = 110;
    try { lblPreset.graphics.foregroundColor = dlg.graphics.newPen(dlg.graphics.PenType.SOLID_COLOR, cTextPrimary, 1); } catch (e) {}

    var PRESET_DESCRIPTIONS = [
        "Natural Studio: Balanced skin evening with 100% natural pore relief & gentle Dodge/Burn.",
        "Glamour Velvet: Soft beauty smoothing with radiant luminance & high vitality.",
        "Deep Acne Fix: Aggressive neural inpainting for deep acne & redness calming.",
        "High-End Editorial: Sculpted facial micro-contours with 100% micro-pore retention.",
        "Custom Configuration: User-defined fine adjustments from the Precision tab."
    ];

    var ddlPreset = grpPresetRow.add("dropdownlist", undefined, [
        "Natural Studio (Subtle & Balanced)",
        "Glamour Velvet (Smooth Beauty)",
        "Deep Blemish & Acne Fix",
        "High-End Editorial (Pore Preserved)",
        "Custom Configuration"
    ]);
    ddlPreset.selection = 0;
    ddlPreset.preferredSize.width = 310;

    // Dynamic Preset Description Card
    var pnlCard = viewTab1.add("panel", undefined, "");
    pnlCard.orientation = "column";
    pnlCard.alignChildren = ["fill", "center"];
    pnlCard.margins = 6;
    var lblPresetDesc = pnlCard.add("statictext", undefined, PRESET_DESCRIPTIONS[0]);
    try {
        lblPresetDesc.graphics.font = ScriptUI.newFont("Segoe UI", "REGULAR", 9);
        lblPresetDesc.graphics.foregroundColor = dlg.graphics.newPen(dlg.graphics.PenType.SOLID_COLOR, cTextMuted, 1);
    } catch (e) {}

    // Primary Action Button
    var btnFullRetouch = viewTab1.add("button", undefined, "EXECUTE COMPLETE STUDIO RETOUCH");
    btnFullRetouch.preferredSize = [435, 36];

    var lblModules = viewTab1.add("statictext", undefined, "Individual Studio Modules:");
    try { lblModules.graphics.foregroundColor = dlg.graphics.newPen(dlg.graphics.PenType.SOLID_COLOR, cTextMuted, 1); } catch (e) {}

    var grpGrid = viewTab1.add("group");
    grpGrid.orientation = "column";
    grpGrid.alignChildren = ["fill", "top"];
    grpGrid.spacing = 6;

    var grpRow1 = grpGrid.add("group");
    grpRow1.orientation = "row";
    grpRow1.spacing = 8;
    var btnHealOnly = grpRow1.add("button", undefined, "Heal Blemishes");
    var btnSmoothOnly = grpRow1.add("button", undefined, "Smooth Skin");
    btnHealOnly.preferredSize = [212, 30];
    btnSmoothOnly.preferredSize = [212, 30];

    var grpRow2 = grpGrid.add("group");
    grpRow2.orientation = "row";
    grpRow2.spacing = 8;
    var btnLightenOnly = grpRow2.add("button", undefined, "Tone Lighten");
    var btnDbOnly = grpRow2.add("button", undefined, "Dodge && Burn");
    btnLightenOnly.preferredSize = [212, 30];
    btnDbOnly.preferredSize = [212, 30];

    var grpRow3 = grpGrid.add("group");
    grpRow3.orientation = "row";
    grpRow3.spacing = 8;
    var btnEyeTeethOnly = grpRow3.add("button", undefined, "Eyes && Teeth");
    var btnShineOnly = grpRow3.add("button", undefined, "Shine Reducer");
    btnEyeTeethOnly.preferredSize = [212, 30];
    btnShineOnly.preferredSize = [212, 30];

    var btnPreviewMask = viewTab1.add("button", undefined, "Preview Blemish Detection Mask");
    btnPreviewMask.preferredSize = [435, 26];

    // =============================================================
    // TAB 2: PRECISION ADJUSTMENTS (Compact 2-Column Grid Layout)
    // =============================================================
    var viewTab2 = tabFolder.add("tab", undefined, "Precision Adjustments");
    viewTab2.orientation = "row";
    viewTab2.alignChildren = ["fill", "top"];
    viewTab2.spacing = 14;
    viewTab2.margins = 10;

    // Left Column
    var colLeft = viewTab2.add("group");
    colLeft.orientation = "column";
    colLeft.alignChildren = ["fill", "top"];
    colLeft.spacing = 6;
    colLeft.preferredSize.width = 215;

    // Slider 1: Blemish Sensitivity
    var grpSens = colLeft.add("group");
    grpSens.orientation = "row";
    var lblSensT = grpSens.add("statictext", undefined, "Sensitivity:");
    lblSensT.preferredSize.width = 75;
    var lblSens = grpSens.add("statictext", undefined, "40%");
    lblSens.preferredSize.width = 35;
    var sldSens = colLeft.add("slider", undefined, 40, 10, 100);
    sldSens.preferredSize.width = 205;
    sldSens.onChanging = function () { lblSens.text = Math.round(sldSens.value) + "%"; ddlPreset.selection = 4; lblPresetDesc.text = PRESET_DESCRIPTIONS[4]; };

    // Slider 2: Skin Smoothing
    var grpSmooth = colLeft.add("group");
    grpSmooth.orientation = "row";
    var lblSmoothT = grpSmooth.add("statictext", undefined, "Smoothing:");
    lblSmoothT.preferredSize.width = 75;
    var lblSmooth = grpSmooth.add("statictext", undefined, "35%");
    lblSmooth.preferredSize.width = 35;
    var sldSmooth = colLeft.add("slider", undefined, 35, 5, 100);
    sldSmooth.preferredSize.width = 205;
    sldSmooth.onChanging = function () { lblSmooth.text = Math.round(sldSmooth.value) + "%"; ddlPreset.selection = 4; lblPresetDesc.text = PRESET_DESCRIPTIONS[4]; };

    // Slider 3: Texture & Pores
    var grpTex = colLeft.add("group");
    grpTex.orientation = "row";
    var lblTexT = grpTex.add("statictext", undefined, "Pores/Tex:");
    lblTexT.preferredSize.width = 75;
    var lblTex = grpTex.add("statictext", undefined, "45%");
    lblTex.preferredSize.width = 35;
    var sldTex = colLeft.add("slider", undefined, 45, 0, 100);
    sldTex.preferredSize.width = 205;
    sldTex.onChanging = function () { lblTex.text = Math.round(sldTex.value) + "%"; ddlPreset.selection = 4; lblPresetDesc.text = PRESET_DESCRIPTIONS[4]; };

    // Slider 4: Tone Lightening
    var grpStr = colLeft.add("group");
    grpStr.orientation = "row";
    var lblStrT = grpStr.add("statictext", undefined, "Tone Light:");
    lblStrT.preferredSize.width = 75;
    var lblStr = grpStr.add("statictext", undefined, "25%");
    lblStr.preferredSize.width = 35;
    var sldStr = colLeft.add("slider", undefined, 25, 0, 100);
    sldStr.preferredSize.width = 205;
    sldStr.onChanging = function () { lblStr.text = Math.round(sldStr.value) + "%"; ddlPreset.selection = 4; lblPresetDesc.text = PRESET_DESCRIPTIONS[4]; };

    // Right Column
    var colRight = viewTab2.add("group");
    colRight.orientation = "column";
    colRight.alignChildren = ["fill", "top"];
    colRight.spacing = 6;
    colRight.preferredSize.width = 215;

    // Slider 5: AI Dodge & Burn
    var grpDb = colRight.add("group");
    grpDb.orientation = "row";
    var lblDbT = grpDb.add("statictext", undefined, "Dodge/Burn:");
    lblDbT.preferredSize.width = 75;
    var lblDb = grpDb.add("statictext", undefined, "35%");
    lblDb.preferredSize.width = 35;
    var sldDb = colRight.add("slider", undefined, 35, 0, 100);
    sldDb.preferredSize.width = 205;
    sldDb.onChanging = function () { lblDb.text = Math.round(sldDb.value) + "%"; ddlPreset.selection = 4; lblPresetDesc.text = PRESET_DESCRIPTIONS[4]; };

    // Slider 6: Shine Reduction
    var grpShine = colRight.add("group");
    grpShine.orientation = "row";
    var lblShineT = grpShine.add("statictext", undefined, "Shine Reduc:");
    lblShineT.preferredSize.width = 75;
    var lblShine = grpShine.add("statictext", undefined, "30%");
    lblShine.preferredSize.width = 35;
    var sldShine = colRight.add("slider", undefined, 30, 0, 100);
    sldShine.preferredSize.width = 205;
    sldShine.onChanging = function () { lblShine.text = Math.round(sldShine.value) + "%"; ddlPreset.selection = 4; lblPresetDesc.text = PRESET_DESCRIPTIONS[4]; };

    // Slider 7: Eye & Teeth Brightening
    var grpEt = colRight.add("group");
    grpEt.orientation = "row";
    var lblEtT = grpEt.add("statictext", undefined, "Eyes/Teeth:");
    lblEtT.preferredSize.width = 75;
    var lblEt = grpEt.add("statictext", undefined, "30%");
    lblEt.preferredSize.width = 35;
    var sldEt = colRight.add("slider", undefined, 30, 0, 100);
    sldEt.preferredSize.width = 205;
    sldEt.onChanging = function () { lblEt.text = Math.round(sldEt.value) + "%"; ddlPreset.selection = 4; lblPresetDesc.text = PRESET_DESCRIPTIONS[4]; };

    // Pimple Treatment Mode Dropdown
    var grpMode = colRight.add("group");
    grpMode.orientation = "column";
    grpMode.alignChildren = ["fill", "top"];
    var lblMode = grpMode.add("statictext", undefined, "Treatment Mode:");
    try { lblMode.graphics.foregroundColor = dlg.graphics.newPen(dlg.graphics.PenType.SOLID_COLOR, cTextMuted, 1); } catch (e) {}

    var ddlHealMode = grpMode.add("dropdownlist", undefined, [
        "Full AI Inpainting & Pores",
        "Calm Redness & Erythema",
        "Flatten 3D Blemish Bumps"
    ]);
    ddlHealMode.selection = 0;
    ddlHealMode.preferredSize.width = 205;

    // =============================================================
    // TAB 3: ENGINE & CONFIGURATION
    // =============================================================
    var viewTab3 = tabFolder.add("tab", undefined, "Engine && Settings");
    viewTab3.orientation = "column";
    viewTab3.alignChildren = ["fill", "top"];
    viewTab3.spacing = 8;
    viewTab3.margins = 10;

    var grpServerBox = viewTab3.add("group");
    grpServerBox.orientation = "row";
    grpServerBox.alignChildren = ["left", "center"];
    var lblActiveSrv = grpServerBox.add("statictext", undefined, "Backend Host:");
    lblActiveSrv.preferredSize.width = 110;
    try { lblActiveSrv.graphics.foregroundColor = dlg.graphics.newPen(dlg.graphics.PenType.SOLID_COLOR, cTextPrimary, 1); } catch (e) {}

    var txtSrvUrl = grpServerBox.add("statictext", undefined, activeServerUrl);
    txtSrvUrl.preferredSize.width = 200;
    try { txtSrvUrl.graphics.foregroundColor = dlg.graphics.newPen(dlg.graphics.PenType.SOLID_COLOR, cTextMuted, 1); } catch (e) {}

    var btnRefreshSrv = grpServerBox.add("button", undefined, "Check Server");
    btnRefreshSrv.preferredSize = [100, 24];

    var grpKeyRow = viewTab3.add("group");
    grpKeyRow.orientation = "row";
    grpKeyRow.alignChildren = ["left", "center"];
    var lblApiKey = grpKeyRow.add("statictext", undefined, "Gemini API Key:");
    lblApiKey.preferredSize.width = 110;
    try { lblApiKey.graphics.foregroundColor = dlg.graphics.newPen(dlg.graphics.PenType.SOLID_COLOR, cTextPrimary, 1); } catch (e) {}

    var txtApiKey = grpKeyRow.add("edittext", undefined, "");
    txtApiKey.preferredSize.width = 200;
    var btnSaveKey = grpKeyRow.add("button", undefined, "Save Key");
    btnSaveKey.preferredSize = [100, 24];

    var chkGroup = viewTab3.add("checkbox", undefined, "Organize output layers in 'AI Retouch Pro' group");
    chkGroup.value = true;
    try { chkGroup.graphics.foregroundColor = dlg.graphics.newPen(dlg.graphics.PenType.SOLID_COLOR, cTextPrimary, 1); } catch (e) {}

    var chkMultiLayer = viewTab3.add("checkbox", undefined, "Generate separate individual layers for each module in full retouch");
    chkMultiLayer.value = true;
    try { chkMultiLayer.graphics.foregroundColor = dlg.graphics.newPen(dlg.graphics.PenType.SOLID_COLOR, cTextPrimary, 1); } catch (e) {}

    // Load Persisted Config
    var userCfg = loadStoredConfig();
    txtApiKey.text = userCfg.apiKey || "";
    if (userCfg.lastPreset !== undefined && userCfg.lastPreset >= 0 && userCfg.lastPreset < ddlPreset.items.length) {
        ddlPreset.selection = userCfg.lastPreset;
        lblPresetDesc.text = PRESET_DESCRIPTIONS[userCfg.lastPreset];
    } else {
        ddlPreset.selection = 0;
        lblPresetDesc.text = PRESET_DESCRIPTIONS[0];
    }
    if (userCfg.organizeGroup !== undefined) chkGroup.value = userCfg.organizeGroup;
    if (userCfg.multiLayer !== undefined) chkMultiLayer.value = userCfg.multiLayer;

    btnRefreshSrv.onClick = function () {
        var h = findActiveServer();
        txtSrvUrl.text = h.url;
        var sBadge = h.online ? "[ONLINE: " + h.url.replace("http://127.0.0.1:", "PORT ") + (h.cuda ? " - CUDA GPU" : " - CPU") + "]" : "[OFFLINE: START SERVER]";
        lblStatusBadge.text = sBadge;
        try { lblStatusBadge.graphics.foregroundColor = dlg.graphics.newPen(dlg.graphics.PenType.SOLID_COLOR, h.online ? cGreen : cRed, 1); } catch (e) {}
    };

    btnSaveKey.onClick = function () {
        var key = txtApiKey.text.replace(/^\s+|\s+$/g, "");
        userCfg.apiKey = key;
        saveStoredConfig(userCfg);
        if (key.length > 5) {
            alert("Gemini API Key saved successfully to disk and server.\nVision analysis will remain active across sessions.", "AI Key Saved");
        } else {
            alert("Gemini API Key cleared.", "AI Key Updated");
        }
    };

    // Presets Handlers
    ddlPreset.onChange = function () {
        var idx = ddlPreset.selection.index;
        lblPresetDesc.text = PRESET_DESCRIPTIONS[idx];
        if (idx === 0) { // Natural Studio
            ddlHealMode.selection = 0;
            sldSens.value = 40; lblSens.text = "40%";
            sldSmooth.value = 35; lblSmooth.text = "35%";
            sldTex.value = 45; lblTex.text = "45%";
            sldStr.value = 25; lblStr.text = "25%";
            sldDb.value = 35; lblDb.text = "35%";
            sldShine.value = 30; lblShine.text = "30%";
            sldEt.value = 30; lblEt.text = "30%";
        } else if (idx === 1) { // Glamour Velvet
            ddlHealMode.selection = 0;
            sldSens.value = 60; lblSens.text = "60%";
            sldSmooth.value = 65; lblSmooth.text = "65%";
            sldTex.value = 25; lblTex.text = "25%";
            sldStr.value = 45; lblStr.text = "45%";
            sldDb.value = 50; lblDb.text = "50%";
            sldShine.value = 50; lblShine.text = "50%";
            sldEt.value = 50; lblEt.text = "50%";
        } else if (idx === 2) { // Deep Acne Fix
            ddlHealMode.selection = 0;
            sldSens.value = 80; lblSens.text = "80%";
            sldSmooth.value = 45; lblSmooth.text = "45%";
            sldTex.value = 35; lblTex.text = "35%";
            sldStr.value = 20; lblStr.text = "20%";
            sldDb.value = 40; lblDb.text = "40%";
            sldShine.value = 30; lblShine.text = "30%";
            sldEt.value = 25; lblEt.text = "25%";
        } else if (idx === 3) { // High-End Editorial
            ddlHealMode.selection = 0;
            sldSens.value = 45; lblSens.text = "45%";
            sldSmooth.value = 30; lblSmooth.text = "30%";
            sldTex.value = 60; lblTex.text = "60%";
            sldStr.value = 30; lblStr.text = "30%";
            sldDb.value = 60; lblDb.text = "60%";
            sldShine.value = 35; lblShine.text = "35%";
            sldEt.value = 40; lblEt.text = "40%";
        }
    };

    function getSelectedHealMode() {
        var mIdx = ddlHealMode.selection ? ddlHealMode.selection.index : 0;
        if (mIdx === 1) return "calm_redness";
        if (mIdx === 2) return "flatten_bump";
        return "full_inpaint";
    }

    // Progress & Footer Bar
    var pnlProgress = dlg.add("group");
    pnlProgress.orientation = "column";
    pnlProgress.alignChildren = ["fill", "top"];
    pnlProgress.spacing = 4;

    var lblProgress = pnlProgress.add("statictext", undefined, "Status: Ready for studio retouching.");
    try {
        lblProgress.graphics.font = ScriptUI.newFont("Segoe UI", "REGULAR", 10);
        lblProgress.graphics.foregroundColor = dlg.graphics.newPen(dlg.graphics.PenType.SOLID_COLOR, cTextMuted, 1);
    } catch (e) {}

    var prgBar = pnlProgress.add("progressbar", undefined, 0, 100);
    prgBar.preferredSize.height = 6;

    function setProgress(pct, msg) {
        prgBar.value = pct;
        lblProgress.text = "Status: " + msg;
        dlg.update();
        app.refresh();
    }

    var grpBottom = dlg.add("group");
    grpBottom.orientation = "row";
    grpBottom.alignChildren = ["fill", "center"];

    var lblVersion = grpBottom.add("statictext", undefined, "AI Engine v3.0 - Professional Studio Pipeline");
    try {
        lblVersion.graphics.font = ScriptUI.newFont("Segoe UI", "REGULAR", 9);
        lblVersion.graphics.foregroundColor = dlg.graphics.newPen(dlg.graphics.PenType.SOLID_COLOR, [0.45, 0.45, 0.45, 1.0], 1);
    } catch (e) {}

    var btnClose = grpBottom.add("button", undefined, "Close");
    btnClose.preferredSize = [80, 24];
    btnClose.onClick = function () { dlg.close(); };

    // =============================================================
    // EXECUTION HANDLERS
    // =============================================================

    function checkServerBeforeRun() {
        var h = findActiveServer();
        if (!h.online) {
            alert("AI Retouch backend server is offline!\n\nPlease start the server by double-clicking:\nbackend\\run_server.bat\n\nOnce the black server window says 'Uvicorn running', retry.", "AI Server Offline");
            return false;
        }
        return true;
    }

    btnFullRetouch.onClick = function () {
        if (!validateActiveDocument()) return;
        if (!checkServerBeforeRun()) return;

        var sens = (Math.round(sldSens.value) / 100.0).toFixed(2);
        var tex = (Math.round(sldTex.value) / 100.0).toFixed(2);
        var smoothStr = (Math.round(sldSmooth.value) / 100.0).toFixed(2);
        var lightStr = (Math.round(sldStr.value) / 100.0).toFixed(2);
        var dbStr = (Math.round(sldDb.value) / 100.0).toFixed(2);
        var shineStr = (Math.round(sldShine.value) / 100.0).toFixed(2);
        var etStr = (Math.round(sldEt.value) / 100.0).toFixed(2);
        var apiKey = txtApiKey.text.replace(/^\s+|\s+$/g, "");
        var healMode = getSelectedHealMode();
        var useSeparateLayers = chkMultiLayer.value;

        userCfg.apiKey = apiKey;
        userCfg.lastPreset = ddlPreset.selection ? ddlPreset.selection.index : 0;
        userCfg.organizeGroup = chkGroup.value;
        userCfg.multiLayer = chkMultiLayer.value;
        saveStoredConfig(userCfg);

        btnFullRetouch.enabled = false;

        app.activeDocument.suspendHistory("Complete Studio Retouch", function () {
            var group = chkGroup.value ? getOrCreateRetouchGroup() : null;

            if (!useSeparateLayers) {
                // High-performance single-pass unified pipeline
                setProgress(30, "Executing unified AI retouch pipeline...");
                executeMasterSuite({
                    includeHeal: true,
                    sensitivity: sens,
                    healMode: healMode,
                    textureBlend: tex,
                    includeDb: (parseFloat(dbStr) > 0.05),
                    dbStrength: dbStr,
                    includeSmooth: (parseFloat(smoothStr) > 0.05),
                    smoothStrength: smoothStr,
                    textureKeep: tex,
                    includeLighten: (parseFloat(lightStr) > 0.05),
                    lightenStrength: lightStr,
                    includeEyeTeeth: (parseFloat(etStr) > 0.05),
                    etStrength: etStr,
                    includeShine: (parseFloat(shineStr) > 0.05),
                    shineStrength: shineStr,
                    apiKey: apiKey
                }, group);
                setProgress(100, "Master retouch completed.");
            } else {
                // Multi-layer staged pipeline
                setProgress(15, "1/5 Segmenting skin and inpainting blemishes...");
                executeAutoHeal(sens, tex, 3, 100, apiKey, group, healMode);

                if (parseFloat(dbStr) > 0.05) {
                    setProgress(40, "2/5 Calculating AI Dodge & Burn contours...");
                    executeDodgeAndBurn(dbStr, 0.6, 4, 85, group);
                }

                setProgress(65, "3/5 Applying frequency separation skin smoothing...");
                executeSmoothSkin(smoothStr, tex, 4, 85, group);

                if (parseFloat(lightStr) > 0.05) {
                    setProgress(80, "4/5 Enhancing facial tone lighting...");
                    executeLightenSkin(lightStr, 4, 90, group);
                }

                if (parseFloat(etStr) > 0.05) {
                    setProgress(90, "5/5 Brightening eyes and whitening teeth...");
                    executeEyeAndTeeth(etStr, etStr, 0.35, 3, 90, group);
                }

                setProgress(100, "Studio retouching completed successfully.");
            }
        });

        btnFullRetouch.enabled = true;
        dlg.close();
    };

    btnHealOnly.onClick = function () {
        if (!validateActiveDocument()) return;
        if (!checkServerBeforeRun()) return;

        var sens = (Math.round(sldSens.value) / 100.0).toFixed(2);
        var tex = (Math.round(sldTex.value) / 100.0).toFixed(2);
        var apiKey = txtApiKey.text.replace(/^\s+|\s+$/g, "");
        var healMode = getSelectedHealMode();

        setProgress(30, "Inpainting detected blemishes...");
        app.activeDocument.suspendHistory("AI Heal Blemishes", function () {
            var group = chkGroup.value ? getOrCreateRetouchGroup() : null;
            executeAutoHeal(sens, tex, 3, 100, apiKey, group, healMode);
            setProgress(100, "Blemishes healed.");
        });
        dlg.close();
    };

    btnSmoothOnly.onClick = function () {
        if (!validateActiveDocument()) return;
        if (!checkServerBeforeRun()) return;

        var smoothStr = (Math.round(sldSmooth.value) / 100.0).toFixed(2);
        var tex = (Math.round(sldTex.value) / 100.0).toFixed(2);

        setProgress(40, "Applying frequency separation smoothing...");
        app.activeDocument.suspendHistory("AI Smooth Skin", function () {
            var group = chkGroup.value ? getOrCreateRetouchGroup() : null;
            executeSmoothSkin(smoothStr, tex, 4, 100, group);
            setProgress(100, "Skin smoothing complete.");
        });
        dlg.close();
    };

    btnLightenOnly.onClick = function () {
        if (!validateActiveDocument()) return;
        if (!checkServerBeforeRun()) return;

        var lightStr = (Math.round(sldStr.value) / 100.0).toFixed(2);

        setProgress(40, "Lightening facial skin tone...");
        app.activeDocument.suspendHistory("AI Tone Lighten", function () {
            var group = chkGroup.value ? getOrCreateRetouchGroup() : null;
            executeLightenSkin(lightStr, 4, 100, group);
            setProgress(100, "Tone lightening complete.");
        });
        dlg.close();
    };

    btnDbOnly.onClick = function () {
        if (!validateActiveDocument()) return;
        if (!checkServerBeforeRun()) return;

        var dbStr = (Math.round(sldDb.value) / 100.0).toFixed(2);

        setProgress(40, "Calculating AI Dodge & Burn...");
        app.activeDocument.suspendHistory("AI Dodge & Burn", function () {
            var group = chkGroup.value ? getOrCreateRetouchGroup() : null;
            executeDodgeAndBurn(dbStr, 0.6, 4, 100, group);
            setProgress(100, "Dodge & Burn complete.");
        });
        dlg.close();
    };

    btnEyeTeethOnly.onClick = function () {
        if (!validateActiveDocument()) return;
        if (!checkServerBeforeRun()) return;

        var etStr = (Math.round(sldEt.value) / 100.0).toFixed(2);

        setProgress(40, "Enhancing eyes and teeth...");
        app.activeDocument.suspendHistory("AI Eye & Teeth Enhancement", function () {
            var group = chkGroup.value ? getOrCreateRetouchGroup() : null;
            executeEyeAndTeeth(etStr, etStr, 0.35, 3, 100, group);
            setProgress(100, "Eyes and teeth enhanced.");
        });
        dlg.close();
    };

    btnShineOnly.onClick = function () {
        if (!validateActiveDocument()) return;
        if (!checkServerBeforeRun()) return;

        var shineStr = (Math.round(sldShine.value) / 100.0).toFixed(2);

        setProgress(40, "Neutralizing specular shine...");
        app.activeDocument.suspendHistory("AI Shine Neutralizer", function () {
            var group = chkGroup.value ? getOrCreateRetouchGroup() : null;
            executeShineNeutralize(shineStr, 0.75, 4, 100, group);
            setProgress(100, "Skin shine neutralized.");
        });
        dlg.close();
    };

    btnPreviewMask.onClick = function () {
        if (!validateActiveDocument()) return;
        if (!checkServerBeforeRun()) return;

        var sens = (Math.round(sldSens.value) / 100.0).toFixed(2);
        var apiKey = txtApiKey.text.replace(/^\s+|\s+$/g, "");
        var group = chkGroup.value ? getOrCreateRetouchGroup() : null;

        setProgress(50, "Generating blemish detection mask preview...");
        previewDetectedMask(sens, apiKey, group);
        dlg.close();
    };

    dlg.center();
    dlg.show();
})();



