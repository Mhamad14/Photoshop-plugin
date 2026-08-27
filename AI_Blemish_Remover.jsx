#target photoshop

/**
 * AI Retouch v2 - Auto-Detection & Dual Action Architecture
 * BiSeNet Skin Segmentation, Hybrid Pimple Detection, Tone-Relative Skin Lightening & Simple-LaMa Inpainting
 * Compatible with Photoshop CS6, CC 2018 - 2026
 */

(function () {
    var SERVER_CANDIDATES = [
        "http://127.0.0.1:8001",
        "http://127.0.0.1:8008",
        "http://127.0.0.1:8000",
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

    // -------------------------------------------------------------
    // 1. AUTO-HEAL: Automated AI Blemish Removal (Action 1)
    // -------------------------------------------------------------
    function executeAutoHeal(sensitivity, textureBlend, featherRadius, opacityVal, apiKey) {
        if (app.documents.length === 0) {
            alert("Please open a portrait photo first!", "AI Retouch");
            return false;
        }

        var doc = app.activeDocument;
        var fullExportFile = new File(tempFolder.fsName + "/full_portrait.png");
        var autoHealedFile = new File(tempFolder.fsName + "/auto_healed.png");

        if (fullExportFile.exists) fullExportFile.remove();
        if (autoHealedFile.exists) autoHealedFile.remove();

        exportCleanPortrait(fullExportFile);

        // Call FastAPI /apply-heal with auto detection
        var cmd = 'curl.exe -s -X POST ' +
            '-F "image=@' + fullExportFile.fsName + '" ' +
            '-F "sensitivity=' + sensitivity + '" ' +
            '-F "texture_blend=' + textureBlend + '" ' +
            '-F "feather_radius=' + featherRadius + '" ' +
            '-F "gemini_api_key=' + apiKey + '" ' +
            '-F "grain_intensity=0.03" ' +
            '-o "' + autoHealedFile.fsName + '" ' +
            '"' + activeServerUrl + '/auto-heal"';

        app.system(cmd);

        if (!autoHealedFile.exists || autoHealedFile.length < 100) {
            alert("Auto-heal failed. Is the server running on " + activeServerUrl + "?\n\nRun 'run_server.bat' in the backend folder.", "AI Server Error");
            return false;
        }

        placeFile(autoHealedFile);

        var placedLayer = doc.activeLayer;
        placedLayer.name = "AI Healed Blemishes";
        placedLayer.opacity = opacityVal;

        return true;
    }

    // -------------------------------------------------------------
    // 2. LIGHTEN SKIN: Tone-Relative Facial Brightening (Action 2)
    // -------------------------------------------------------------
    function executeLightenSkin(strength, featherRadius, opacityVal) {
        if (app.documents.length === 0) {
            alert("Please open a portrait photo first!", "AI Retouch");
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

        if (!lightenedFile.exists || lightenedFile.length < 100) {
            alert("Skin lightening failed. Is the server running on " + activeServerUrl + "?", "AI Server Error");
            return false;
        }

        placeFile(lightenedFile);

        var placedLayer = doc.activeLayer;
        placedLayer.name = "AI Lightened Skin";
        placedLayer.opacity = opacityVal;

        return true;
    }

    // -------------------------------------------------------------
    // 3. DODGE & BURN: AI Micro-Contrast Skin Evening
    // -------------------------------------------------------------
    function executeDodgeBurn(strength, featherRadius, opacityVal) {
        if (app.documents.length === 0) {
            alert("Please open a portrait photo first!", "AI Retouch");
            return false;
        }

        var doc = app.activeDocument;
        var fullExportFile = new File(tempFolder.fsName + "/full_portrait.png");
        var dbFile = new File(tempFolder.fsName + "/dodge_burn.png");

        if (fullExportFile.exists) fullExportFile.remove();
        if (dbFile.exists) dbFile.remove();

        exportCleanPortrait(fullExportFile);

        var cmd = 'curl.exe -s -X POST ' +
            '-F "image=@' + fullExportFile.fsName + '" ' +
            '-F "strength=' + strength + '" ' +
            '-F "feather_radius=' + featherRadius + '" ' +
            '-o "' + dbFile.fsName + '" ' +
            '"' + activeServerUrl + '/apply-dodge-burn"';

        app.system(cmd);

        if (!dbFile.exists || dbFile.length < 100) {
            alert("Dodge & Burn failed. Is the server running on " + activeServerUrl + "?", "AI Server Error");
            return false;
        }

        placeFile(dbFile);

        var placedLayer = doc.activeLayer;
        placedLayer.name = "AI Dodge & Burn";
        placedLayer.opacity = opacityVal;

        return true;
    }

    // -------------------------------------------------------------
    // 4. PREVIEW DETECTED MASKS
    // -------------------------------------------------------------
    function previewDetectedMask(sensitivity, apiKey) {
        if (app.documents.length === 0) {
            alert("Please open a photo first!", "AI Retouch");
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
            return true;
        }
        return false;
    }

    // -------------------------------------------------------------
    // UI Dialog Window
    // -------------------------------------------------------------
    var dlg = new Window("dialog", "AI Retouch Studio (Pro Retouch Suite)", undefined);
    dlg.orientation = "column";
    dlg.alignChildren = ["fill", "top"];
    dlg.spacing = 8;
    dlg.margins = 14;

    // Server Status
    var health = findActiveServer();
    var pnlStatus = dlg.add("panel", undefined, "Backend & AI Vision Status");
    pnlStatus.orientation = "column";
    pnlStatus.alignChildren = ["fill", "top"];
    pnlStatus.spacing = 4;

    var grpStatusRow = pnlStatus.add("group");
    grpStatusRow.orientation = "row";
    var txtStatus = grpStatusRow.add("statictext", undefined, health.online ? "● Online (" + health.url.replace("http://127.0.0.1:", "Port ") + ")" : "● Offline (Start run_server.bat)");
    txtStatus.preferredSize.width = 170;

    var btnRefresh = grpStatusRow.add("button", undefined, "Check");
    btnRefresh.preferredSize.width = 50;

    // Gemini API Key row
    var grpKey = pnlStatus.add("group");
    grpKey.orientation = "row";
    grpKey.add("statictext", undefined, "Gemini Key:");
    var txtApiKey = grpKey.add("edittext", undefined, "");
    txtApiKey.preferredSize.width = 140;
    var btnSaveKey = grpKey.add("button", undefined, "Save");
    btnSaveKey.preferredSize.width = 55;

    btnSaveKey.onClick = function () {
        var key = txtApiKey.text.replace(/^\s+|\s+$/g, "");
        if (key.length > 5) {
            if (saveApiKeyToServer(key)) {
                alert("Gemini API Key saved! Gemini Vision AI active.", "Gemini AI Ready");
                txtStatus.text = "● Online (Gemini Vision Active)";
            }
        }
    };

    btnRefresh.onClick = function () {
        var h = findActiveServer();
        txtStatus.text = h.online ? "● Online (" + h.url.replace("http://127.0.0.1:", "Port ") + (h.gemini ? " - Gemini Active" : "") + ")" : "● Offline (Start run_server.bat)";
    };

    // --- SECTION: ACTIONS ---
    var pnlAuto = dlg.add("panel", undefined, "Automated Retouch Actions");
    pnlAuto.orientation = "column";
    pnlAuto.alignChildren = ["fill", "top"];
    pnlAuto.spacing = 6;

    var btnDualRetouch = pnlAuto.add("button", undefined, "1. Full Studio Retouch (Heal + D&B + Tone)");
    btnDualRetouch.preferredSize.height = 36;

    var grpSplitBtns = pnlAuto.add("group");
    grpSplitBtns.orientation = "row";
    var btnHealOnly = grpSplitBtns.add("button", undefined, "Remove Pimples");
    btnHealOnly.preferredSize.width = 95;
    btnHealOnly.preferredSize.height = 28;

    var btnDbOnly = grpSplitBtns.add("button", undefined, "Dodge & Burn");
    btnDbOnly.preferredSize.width = 95;
    btnDbOnly.preferredSize.height = 28;

    var btnLightenOnly = grpSplitBtns.add("button", undefined, "Tone Lift");
    btnLightenOnly.preferredSize.width = 95;
    btnLightenOnly.preferredSize.height = 28;

    var btnPreviewMask = pnlAuto.add("button", undefined, "Preview Mask Overlay");
    btnPreviewMask.preferredSize.height = 24;

    // --- SECTION: PARAMETERS ---
    var pnlSettings = dlg.add("panel", undefined, "Controls & Parameters");
    pnlSettings.orientation = "column";
    pnlSettings.alignChildren = ["fill", "top"];
    pnlSettings.spacing = 5;

    // Pimple Sensitivity Slider
    var grpSens = pnlSettings.add("group");
    grpSens.orientation = "row";
    grpSens.add("statictext", undefined, "Pimple Sensitivity:");
    var lblSens = grpSens.add("statictext", undefined, "50%");
    lblSens.preferredSize.width = 35;
    var sldSens = pnlSettings.add("slider", undefined, 50, 10, 100);
    sldSens.onChanging = function () { lblSens.text = Math.round(sldSens.value) + "%"; };

    // Skin Lightening Strength
    var grpStr = pnlSettings.add("group");
    grpStr.orientation = "row";
    grpStr.add("statictext", undefined, "Skin Tone Strength:");
    var lblStr = grpStr.add("statictext", undefined, "35%");
    lblStr.preferredSize.width = 35;
    var sldStr = pnlSettings.add("slider", undefined, 35, 5, 100);
    sldStr.onChanging = function () { lblStr.text = Math.round(sldStr.value) + "%"; };

    // Skin Texture Blend
    var grpTex = pnlSettings.add("group");
    grpTex.orientation = "row";
    grpTex.add("statictext", undefined, "Skin Texture & Pores:");
    var lblTex = grpTex.add("statictext", undefined, "25%");
    lblTex.preferredSize.width = 35;
    var sldTex = pnlSettings.add("slider", undefined, 25, 0, 100);
    sldTex.onChanging = function () { lblTex.text = Math.round(sldTex.value) + "%"; };

    // --- EVENT HANDLERS ---
    btnDualRetouch.onClick = function () {
        var sens = (Math.round(sldSens.value) / 100.0).toFixed(2);
        var tex = (Math.round(sldTex.value) / 100.0).toFixed(2);
        var str = (Math.round(sldStr.value) / 100.0).toFixed(2);
        var apiKey = txtApiKey.text.replace(/^\s+|\s+$/g, "");

        btnDualRetouch.text = "Executing AI Retouching...";
        btnDualRetouch.enabled = false;
        dlg.update();

        // 1. Remove Pimples First
        var hOk = executeAutoHeal(sens, tex, 3, 100, apiKey);
        // 2. Lighten Skin Second
        var lOk = executeLightenSkin(str, 4, 100);

        if (hOk || lOk) {
            dlg.close();
        } else {
            btnDualRetouch.text = "1. Full Auto-Retouch (Heal + Tone Balance)";
            btnDualRetouch.enabled = true;
            dlg.update();
        }
    };

    btnHealOnly.onClick = function () {
        var sens = (Math.round(sldSens.value) / 100.0).toFixed(2);
        var tex = (Math.round(sldTex.value) / 100.0).toFixed(2);
        var apiKey = txtApiKey.text.replace(/^\s+|\s+$/g, "");

        btnHealOnly.text = "Healing...";
        btnHealOnly.enabled = false;
        dlg.update();

        if (executeAutoHeal(sens, tex, 3, 100, apiKey)) {
            dlg.close();
        } else {
            btnHealOnly.text = "Remove Pimples";
            btnHealOnly.enabled = true;
            dlg.update();
        }
    };

    btnDbOnly.onClick = function () {
        var str = (Math.round(sldStr.value) / 100.0).toFixed(2);

        btnDbOnly.text = "Processing D&B...";
        btnDbOnly.enabled = false;
        dlg.update();

        if (executeDodgeBurn(str, 4, 100)) {
            dlg.close();
        } else {
            btnDbOnly.text = "Dodge & Burn";
            btnDbOnly.enabled = true;
            dlg.update();
        }
    };

    btnLightenOnly.onClick = function () {
        var str = (Math.round(sldStr.value) / 100.0).toFixed(2);

        btnLightenOnly.text = "Processing...";
        btnLightenOnly.enabled = false;
        dlg.update();

        if (executeLightenSkin(str, 4, 100)) {
            dlg.close();
        } else {
            btnLightenOnly.text = "Tone Lift";
            btnLightenOnly.enabled = true;
            dlg.update();
        }
    };

    btnPreviewMask.onClick = function () {
        var sens = (Math.round(sldSens.value) / 100.0).toFixed(2);
        var apiKey = txtApiKey.text.replace(/^\s+|\s+$/g, "");
        previewDetectedMask(sens, apiKey);
        dlg.close();
    };

    var btnClose = dlg.add("button", undefined, "Close");
    btnClose.onClick = function () { dlg.close(); };

    dlg.center();
    dlg.show();
})();
