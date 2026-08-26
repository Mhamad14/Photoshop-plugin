#target photoshop

/**
 * AI Retouch Pro v2.5 - Professional Automated Studio Retouching
 * BiSeNet Skin Segmentation, Multi-Scale Acne & Blemish Detection,
 * Edge-Preserving Frequency Separation & Tone-Relative Lightening.
 * Compatible with Photoshop CS6, CC 2018 - 2026.
 */

(function () {
    // Priority candidate URLs for local AI server
    var SERVER_CANDIDATES = [
        "http://127.0.0.1:8001",
        "http://127.0.0.1:8765",
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
    // Health Check & Port Discovery
    // -------------------------------------------------------------
    function findActiveServer() {
        for (var i = 0; i < SERVER_CANDIDATES.length; i++) {
            var url = SERVER_CANDIDATES[i];
            var testFile = new File(tempFolder.fsName + "/health_" + i + ".json");
            if (testFile.exists) testFile.remove();

            var cmd = 'curl.exe -s -m 2 "' + url + '/health" -o "' + testFile.fsName + '"';
            app.system(cmd);

            if (testFile.exists && testFile.length > 5) {
                testFile.open("r");
                var content = testFile.read();
                testFile.close();
                testFile.remove();
                if (content.indexOf("ready") !== -1 || content.indexOf("status") !== -1) {
                    activeServerUrl = url;
                    var hasGemini = (content.indexOf('"gemini_enabled": true') !== -1 || content.indexOf('"gemini_enabled":true') !== -1);
                    return { online: true, url: url, gemini: hasGemini, info: content };
                }
            }
        }
        return { online: false, url: activeServerUrl, gemini: false, info: "Server offline" };
    }

    function saveApiKeyToServer(key) {
        var respFile = new File(tempFolder.fsName + "/set_key_resp.json");
        if (respFile.exists) respFile.remove();

        var cmd = 'curl.exe -s -X POST -F "gemini_api_key=' + key + '" -o "' + respFile.fsName + '" "' + activeServerUrl + '/set-api-key"';
        app.system(cmd);

        if (respFile.exists) {
            respFile.remove();
            return true;
        }
        return false;
    }

    function placeFile(file) {
        var desc = new ActionDescriptor();
        desc.putPath(charIDToTypeID("null"), file);
        desc.putEnumerated(charIDToTypeID("FTcs"), charIDToTypeID("QCSt"), charIDToTypeID("Qcsa"));
        var offsetDesc = new ActionDescriptor();
        offsetDesc.putUnitDouble(charIDToTypeID("Hrzn"), charIDToTypeID("#Pxl"), 0.0);
        offsetDesc.putUnitDouble(charIDToTypeID("Vrtc"), charIDToTypeID("#Pxl"), 0.0);
        desc.putObject(charIDToTypeID("Ofst"), charIDToTypeID("Ofst"), offsetDesc);
        executeAction(charIDToTypeID("Plc "), desc, DialogModes.NO);
    }

    function exportCleanPortrait(targetFile) {
        var doc = app.activeDocument;
        var pngSaveOptions = new PNGSaveOptions();
        pngSaveOptions.compression = 0;
        pngSaveOptions.interlaced = false;

        var dupDoc = doc.duplicate("AI_Temp_Export", false);
        for (var k = 0; k < dupDoc.artLayers.length; k++) {
            var l = dupDoc.artLayers[k];
            if (!l.isBackgroundLayer && (l.name.indexOf("AI ") === 0 || l.name.indexOf("Mask") !== -1)) {
                l.visible = false;
            }
        }
        dupDoc.flatten();
        dupDoc.saveAs(targetFile, pngSaveOptions, true, Extension.LOWERCASE);
        dupDoc.close(SaveOptions.DONOTSAVECHANGES);
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

    // -------------------------------------------------------------
    // 1. AUTO-HEAL: Automated AI Blemish & Acne Removal
    // -------------------------------------------------------------
    function executeAutoHeal(sensitivity, textureBlend, featherRadius, opacityVal, apiKey, targetGroup, healMode) {
        healMode = healMode || "full_inpaint";
        if (app.documents.length === 0) {
            alert("Please open a portrait photo first.", "AI Retouch");
            return false;
        }

        var doc = app.activeDocument;
        var fullExportFile = new File(tempFolder.fsName + "/full_portrait.png");
        var autoHealedFile = new File(tempFolder.fsName + "/auto_healed.png");

        if (fullExportFile.exists) fullExportFile.remove();
        if (autoHealedFile.exists) autoHealedFile.remove();

        exportCleanPortrait(fullExportFile);

        var cmd = 'curl.exe -s -X POST ' +
            '-F "image=@' + fullExportFile.fsName + '" ' +
            '-F "sensitivity=' + sensitivity + '" ' +
            '-F "heal_mode=' + healMode + '" ' +
            '-F "texture_blend=' + textureBlend + '" ' +
            '-F "feather_radius=' + featherRadius + '" ' +
            '-F "gemini_api_key=' + apiKey + '" ' +
            '-F "grain_intensity=0.03" ' +
            '-o "' + autoHealedFile.fsName + '" ' +
            '"' + activeServerUrl + '/auto-heal"';

        app.system(cmd);

        if (!autoHealedFile.exists || autoHealedFile.length < 500) {
            alert("Auto-heal failed. Verify that the AI server is active on " + activeServerUrl + "\n\nRun backend\\run_server.bat to start the AI server.", "AI Server Error");
            return false;
        }

        placeFile(autoHealedFile);

        var placedLayer = doc.activeLayer;
        placedLayer.name = healMode === "calm_redness" ? "1. AI Calmed Redness" : (healMode === "flatten_bump" ? "1. AI Flattened Blemishes" : "1. AI Healed Blemishes");
        placedLayer.opacity = opacityVal;

        if (targetGroup) {
            try { placedLayer.move(targetGroup, ElementPlacement.INSIDE); } catch (e) {}
        }

        return true;
    }

    // -------------------------------------------------------------
    // 2. SMOOTH SKIN: Frequency Separation & Redness Evening
    // -------------------------------------------------------------
    function executeSmoothSkin(strength, textureKeep, featherRadius, opacityVal, targetGroup) {
        if (app.documents.length === 0) {
            alert("Please open a portrait photo first.", "AI Retouch");
            return false;
        }

        var doc = app.activeDocument;
        var fullExportFile = new File(tempFolder.fsName + "/full_portrait.png");
        var smoothedFile = new File(tempFolder.fsName + "/smoothed_skin.png");

        if (fullExportFile.exists) fullExportFile.remove();
        if (smoothedFile.exists) smoothedFile.remove();

        exportCleanPortrait(fullExportFile);

        var cmd = 'curl.exe -s -X POST ' +
            '-F "image=@' + fullExportFile.fsName + '" ' +
            '-F "strength=' + strength + '" ' +
            '-F "texture_keep=' + textureKeep + '" ' +
            '-F "feather_radius=' + featherRadius + '" ' +
            '-o "' + smoothedFile.fsName + '" ' +
            '"' + activeServerUrl + '/apply-smooth"';

        app.system(cmd);

        if (!smoothedFile.exists || smoothedFile.length < 500) {
            alert("Skin smoothing failed. Verify that the AI server is active on " + activeServerUrl, "AI Server Error");
            return false;
        }

        placeFile(smoothedFile);

        var placedLayer = doc.activeLayer;
        placedLayer.name = "2. AI Smoothed Skin";
        placedLayer.opacity = opacityVal;

        if (targetGroup) {
            try { placedLayer.move(targetGroup, ElementPlacement.INSIDE); } catch (e) {}
        }

        return true;
    }

    // -------------------------------------------------------------
    // 3. LIGHTEN SKIN: Tone-Relative Facial Brightening
    // -------------------------------------------------------------
    function executeLightenSkin(strength, featherRadius, opacityVal, targetGroup) {
        if (app.documents.length === 0) {
            alert("Please open a portrait photo first.", "AI Retouch");
            return false;
        }

        var doc = app.activeDocument;
        var fullExportFile = new File(tempFolder.fsName + "/full_portrait.png");
        var lightenedFile = new File(tempFolder.fsName + "/lightened_skin.png");

        if (fullExportFile.exists) fullExportFile.remove();
        if (lightenedFile.exists) lightenedFile.remove();

        exportCleanPortrait(fullExportFile);

        var cmd = 'curl.exe -s -X POST ' +
            '-F "image=@' + fullExportFile.fsName + '" ' +
            '-F "strength=' + strength + '" ' +
            '-F "feather_radius=' + featherRadius + '" ' +
            '-o "' + lightenedFile.fsName + '" ' +
            '"' + activeServerUrl + '/apply-lighten"';

        app.system(cmd);

        if (!lightenedFile.exists || lightenedFile.length < 500) {
            alert("Skin lightening failed. Verify that the AI server is active on " + activeServerUrl, "AI Server Error");
            return false;
        }

        placeFile(lightenedFile);

        var placedLayer = doc.activeLayer;
        placedLayer.name = "3. AI Lightened Skin";
        placedLayer.opacity = opacityVal;

        if (targetGroup) {
            try { placedLayer.move(targetGroup, ElementPlacement.INSIDE); } catch (e) {}
        }

        return true;
    }

    // -------------------------------------------------------------
    // 4. PREVIEW DETECTED MASKS
    // -------------------------------------------------------------
    function previewDetectedMask(sensitivity, apiKey, targetGroup) {
        if (app.documents.length === 0) {
            alert("Please open a photo first.", "AI Retouch");
            return false;
        }

        var doc = app.activeDocument;
        var fullExportFile = new File(tempFolder.fsName + "/full_portrait.png");
        var maskPreviewFile = new File(tempFolder.fsName + "/detected_mask.png");

        if (fullExportFile.exists) fullExportFile.remove();
        if (maskPreviewFile.exists) maskPreviewFile.remove();

        exportCleanPortrait(fullExportFile);

        var cmd = 'curl.exe -s -X POST ' +
            '-F "image=@' + fullExportFile.fsName + '" ' +
            '-F "sensitivity=' + sensitivity + '" ' +
            '-F "gemini_api_key=' + apiKey + '" ' +
            '-o "' + maskPreviewFile.fsName + '" ' +
            '"' + activeServerUrl + '/detect-mask"';

        app.system(cmd);

        if (maskPreviewFile.exists && maskPreviewFile.length > 100) {
            placeFile(maskPreviewFile);
            var placedLayer = doc.activeLayer;
            placedLayer.name = "AI Detected Blemish Mask (Preview)";
            placedLayer.opacity = 80;
            if (targetGroup) {
                try { placedLayer.move(targetGroup, ElementPlacement.INSIDE); } catch (e) {}
            }
            return true;
        }
        return false;
    }

    // -------------------------------------------------------------
    // UI Dialog Window
    // -------------------------------------------------------------
    var dlg = new Window("dialog", "AI Retouch Pro - Studio Suite");
    dlg.orientation = "column";
    dlg.alignChildren = ["fill", "top"];
    dlg.spacing = 10;
    dlg.margins = 16;

    // --- PANEL 1: SERVER & STATUS ---
    var health = findActiveServer();
    var pnlStatus = dlg.add("panel", undefined, "AI Engine & Connection");
    pnlStatus.orientation = "column";
    pnlStatus.alignChildren = ["fill", "top"];
    pnlStatus.spacing = 6;
    pnlStatus.margins = 12;

    var grpStatusRow = pnlStatus.add("group");
    grpStatusRow.orientation = "row";
    grpStatusRow.alignChildren = ["left", "center"];
    var statusTextVal = health.online ? "[Online] " + health.url.replace("http://127.0.0.1:", "Port ") + (health.gemini ? " (Gemini Active)" : "") : "[Offline] Start backend server";
    var txtStatus = grpStatusRow.add("statictext", undefined, statusTextVal);
    txtStatus.preferredSize.width = 250;

    var btnRefresh = grpStatusRow.add("button", undefined, "Refresh");
    btnRefresh.preferredSize.width = 70;

    // Gemini API Key row
    var grpKey = pnlStatus.add("group");
    grpKey.orientation = "row";
    grpKey.alignChildren = ["left", "center"];
    grpKey.add("statictext", undefined, "Gemini API Key:");
    var txtApiKey = grpKey.add("edittext", undefined, "");
    txtApiKey.preferredSize.width = 170;
    var btnSaveKey = grpKey.add("button", undefined, "Save");
    btnSaveKey.preferredSize.width = 70;

    btnSaveKey.onClick = function () {
        var key = txtApiKey.text.replace(/^\s+|\s+$/g, "");
        if (key.length > 5) {
            if (saveApiKeyToServer(key)) {
                alert("Gemini API Key saved successfully. Vision analysis is active.", "Gemini Ready");
                txtStatus.text = "[Online] Gemini Vision Active";
            }
        }
    };

    btnRefresh.onClick = function () {
        var h = findActiveServer();
        txtStatus.text = h.online ? "[Online] " + h.url.replace("http://127.0.0.1:", "Port ") + (h.gemini ? " (Gemini Active)" : "") : "[Offline] Start backend server";
    };

    // --- PANEL 2: PRESETS & RETOUCH ACTIONS ---
    var pnlPresets = dlg.add("panel", undefined, "Presets & Studio Actions");
    pnlPresets.orientation = "column";
    pnlPresets.alignChildren = ["fill", "top"];
    pnlPresets.spacing = 8;
    pnlPresets.margins = 12;

    var grpPresetRow = pnlPresets.add("group");
    grpPresetRow.orientation = "row";
    grpPresetRow.alignChildren = ["left", "center"];
    grpPresetRow.add("statictext", undefined, "Retouch Preset:");
    var ddlPreset = grpPresetRow.add("dropdownlist", undefined, [
        "Natural Studio (Subtle & Clean)",
        "Glamour & Beauty (Velvety Skin)",
        "Acne & Blemish Fix (Deep Heal)",
        "High-End Editorial (Pore-Preserved)",
        "Custom Configuration"
    ]);
    ddlPreset.selection = 0;
    ddlPreset.preferredSize.width = 240;

    var btnFullRetouch = pnlPresets.add("button", undefined, "Execute Complete Retouch (Heal + Smooth + Lighten)");
    btnFullRetouch.preferredSize.height = 36;

    var grpSplitBtns = pnlPresets.add("group");
    grpSplitBtns.orientation = "row";
    grpSplitBtns.spacing = 6;
    var btnHealOnly = grpSplitBtns.add("button", undefined, "Remove Blemishes");
    btnHealOnly.preferredSize.width = 110;
    btnHealOnly.preferredSize.height = 28;

    var btnSmoothOnly = grpSplitBtns.add("button", undefined, "Smooth Skin");
    btnSmoothOnly.preferredSize.width = 100;
    btnSmoothOnly.preferredSize.height = 28;

    var btnLightenOnly = grpSplitBtns.add("button", undefined, "Lighten Skin");
    btnLightenOnly.preferredSize.width = 100;
    btnLightenOnly.preferredSize.height = 28;

    var btnPreviewMask = pnlPresets.add("button", undefined, "Preview Blemish Detection Mask");
    btnPreviewMask.preferredSize.height = 24;

    // --- PANEL 3: FINE-TUNING CONTROLS ---
    var pnlSettings = dlg.add("panel", undefined, "Precision Parameters");
    pnlSettings.orientation = "column";
    pnlSettings.alignChildren = ["fill", "top"];
    pnlSettings.spacing = 6;
    pnlSettings.margins = 12;

    // Blemish Treatment Mode
    var grpMode = pnlSettings.add("group");
    grpMode.orientation = "row";
    grpMode.alignChildren = ["left", "center"];
    grpMode.add("statictext", undefined, "Pimple Treatment:");
    var ddlHealMode = grpMode.add("dropdownlist", undefined, [
        "Full AI Inpainting & Pore Restoration",
        "Calm Redness & Erythema",
        "Flatten 3D Blemish Bumps"
    ]);
    ddlHealMode.selection = 0;
    ddlHealMode.preferredSize.width = 220;

    // Pimple Sensitivity Slider
    var grpSens = pnlSettings.add("group");
    grpSens.orientation = "row";
    grpSens.add("statictext", undefined, "Blemish Sensitivity:");
    var lblSens = grpSens.add("statictext", undefined, "45%");
    lblSens.preferredSize.width = 40;
    var sldSens = pnlSettings.add("slider", undefined, 45, 10, 100);
    sldSens.onChanging = function () {
        lblSens.text = Math.round(sldSens.value) + "%";
        ddlPreset.selection = 4;
    };

    // Skin Smoothing Strength
    var grpSmooth = pnlSettings.add("group");
    grpSmooth.orientation = "row";
    grpSmooth.add("statictext", undefined, "Skin Smoothing:");
    var lblSmooth = grpSmooth.add("statictext", undefined, "40%");
    lblSmooth.preferredSize.width = 40;
    var sldSmooth = pnlSettings.add("slider", undefined, 40, 5, 100);
    sldSmooth.onChanging = function () {
        lblSmooth.text = Math.round(sldSmooth.value) + "%";
        ddlPreset.selection = 4;
    };

    // Skin Lightening Strength
    var grpStr = pnlSettings.add("group");
    grpStr.orientation = "row";
    grpStr.add("statictext", undefined, "Tone Lightening:");
    var lblStr = grpStr.add("statictext", undefined, "30%");
    lblStr.preferredSize.width = 40;
    var sldStr = pnlSettings.add("slider", undefined, 30, 0, 100);
    sldStr.onChanging = function () {
        lblStr.text = Math.round(sldStr.value) + "%";
        ddlPreset.selection = 4;
    };

    // Skin Texture Blend
    var grpTex = pnlSettings.add("group");
    grpTex.orientation = "row";
    grpTex.add("statictext", undefined, "Texture & Pores:");
    var lblTex = grpTex.add("statictext", undefined, "40%");
    lblTex.preferredSize.width = 40;
    var sldTex = pnlSettings.add("slider", undefined, 40, 0, 100);
    sldTex.onChanging = function () {
        lblTex.text = Math.round(sldTex.value) + "%";
        ddlPreset.selection = 4;
    };

    // Preset selection change handler
    ddlPreset.onChange = function () {
        var idx = ddlPreset.selection.index;
        if (idx === 0) { // Natural Studio
            ddlHealMode.selection = 0;
            sldSens.value = 40; lblSens.text = "40%";
            sldSmooth.value = 35; lblSmooth.text = "35%";
            sldStr.value = 25; lblStr.text = "25%";
            sldTex.value = 45; lblTex.text = "45%";
        } else if (idx === 1) { // Glamour & Beauty
            ddlHealMode.selection = 0;
            sldSens.value = 60; lblSens.text = "60%";
            sldSmooth.value = 65; lblSmooth.text = "65%";
            sldStr.value = 45; lblStr.text = "45%";
            sldTex.value = 25; lblTex.text = "25%";
        } else if (idx === 2) { // Acne & Blemish Fix
            ddlHealMode.selection = 0;
            sldSens.value = 75; lblSens.text = "75%";
            sldSmooth.value = 40; lblSmooth.text = "40%";
            sldStr.value = 20; lblStr.text = "20%";
            sldTex.value = 35; lblTex.text = "35%";
        } else if (idx === 3) { // High-End Editorial
            ddlHealMode.selection = 0;
            sldSens.value = 50; lblSens.text = "50%";
            sldSmooth.value = 45; lblSmooth.text = "45%";
            sldStr.value = 35; lblStr.text = "35%";
            sldTex.value = 55; lblTex.text = "55%";
        }
    };

    function getSelectedHealMode() {
        var mIdx = ddlHealMode.selection ? ddlHealMode.selection.index : 0;
        if (mIdx === 1) return "calm_redness";
        if (mIdx === 2) return "flatten_bump";
        return "full_inpaint";
    }

    // Layer Grouping Checkbox
    var chkGroup = dlg.add("checkbox", undefined, "Organize output layers in 'AI Retouch Pro' group");
    chkGroup.value = true;

    // --- EVENT HANDLERS ---
    btnFullRetouch.onClick = function () {
        if (app.documents.length === 0) {
            alert("Please open a portrait photo first.", "AI Retouch");
            return;
        }

        var sens = (Math.round(sldSens.value) / 100.0).toFixed(2);
        var tex = (Math.round(sldTex.value) / 100.0).toFixed(2);
        var smoothStr = (Math.round(sldSmooth.value) / 100.0).toFixed(2);
        var lightStr = (Math.round(sldStr.value) / 100.0).toFixed(2);
        var apiKey = txtApiKey.text.replace(/^\s+|\s+$/g, "");
        var healMode = getSelectedHealMode();

        btnFullRetouch.text = "Processing Retouching...";
        btnFullRetouch.enabled = false;
        dlg.update();

        app.activeDocument.suspendHistory("Complete AI Retouch", function () {
            var group = chkGroup.value ? getOrCreateRetouchGroup() : null;
            executeAutoHeal(sens, tex, 3, 100, apiKey, group, healMode);
            executeSmoothSkin(smoothStr, tex, 4, 85, group);
            if (parseFloat(lightStr) > 0.05) {
                executeLightenSkin(lightStr, 4, 90, group);
            }
        });

        dlg.close();
    };

    btnHealOnly.onClick = function () {
        if (app.documents.length === 0) {
            alert("Please open a portrait photo first.", "AI Retouch");
            return;
        }
        var sens = (Math.round(sldSens.value) / 100.0).toFixed(2);
        var tex = (Math.round(sldTex.value) / 100.0).toFixed(2);
        var apiKey = txtApiKey.text.replace(/^\s+|\s+$/g, "");
        var healMode = getSelectedHealMode();

        btnHealOnly.text = "Healing...";
        btnHealOnly.enabled = false;
        dlg.update();

        app.activeDocument.suspendHistory("AI Heal Acne & Blemishes", function () {
            var group = chkGroup.value ? getOrCreateRetouchGroup() : null;
            executeAutoHeal(sens, tex, 3, 100, apiKey, group, healMode);
        });

        dlg.close();
    };

    btnSmoothOnly.onClick = function () {
        if (app.documents.length === 0) {
            alert("Please open a portrait photo first.", "AI Retouch");
            return;
        }
        var smoothStr = (Math.round(sldSmooth.value) / 100.0).toFixed(2);
        var tex = (Math.round(sldTex.value) / 100.0).toFixed(2);

        btnSmoothOnly.text = "Smoothing...";
        btnSmoothOnly.enabled = false;
        dlg.update();

        app.activeDocument.suspendHistory("AI Smooth Skin", function () {
            var group = chkGroup.value ? getOrCreateRetouchGroup() : null;
            executeSmoothSkin(smoothStr, tex, 4, 100, group);
        });

        dlg.close();
    };

    btnLightenOnly.onClick = function () {
        if (app.documents.length === 0) {
            alert("Please open a portrait photo first.", "AI Retouch");
            return;
        }
        var lightStr = (Math.round(sldStr.value) / 100.0).toFixed(2);

        btnLightenOnly.text = "Lightening...";
        btnLightenOnly.enabled = false;
        dlg.update();

        app.activeDocument.suspendHistory("AI Lighten Skin", function () {
            var group = chkGroup.value ? getOrCreateRetouchGroup() : null;
            executeLightenSkin(lightStr, 4, 100, group);
        });

        dlg.close();
    };

    btnPreviewMask.onClick = function () {
        var sens = (Math.round(sldSens.value) / 100.0).toFixed(2);
        var apiKey = txtApiKey.text.replace(/^\s+|\s+$/g, "");
        var group = chkGroup.value ? getOrCreateRetouchGroup() : null;
        previewDetectedMask(sens, apiKey, group);
        dlg.close();
    };

    // Bottom Action Row
    var grpBottom = dlg.add("group");
    grpBottom.alignment = ["right", "bottom"];
    var btnClose = grpBottom.add("button", undefined, "Close");
    btnClose.preferredSize.width = 80;
    btnClose.onClick = function () { dlg.close(); };

    dlg.center();
    dlg.show();
})();
