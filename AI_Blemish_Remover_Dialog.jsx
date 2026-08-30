#target photoshop

/**
 * AI Retouch - Blemish & Pimple Remover (Modal Dialog Version)
 * Fast 1-click inpainting in Photoshop CS6 - CC 2026
 */

(function () {
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

    function checkBackendHealth() {
        for (var i = 0; i < SERVER_CANDIDATES.length; i++) {
            var url = SERVER_CANDIDATES[i];
            var testFile = new File(tempFolder.fsName + "/health_test.json");
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
                    return { online: true, url: url, info: content };
                }
            }
        }
        return { online: false, url: activeServerUrl, info: "Server offline" };
    }

    function createMaskLayer() {
        if (app.documents.length === 0) {
            alert("Please open an image in Photoshop first.", "AI Retouch");
            return;
        }
        var doc = app.activeDocument;
        var newLayer = doc.artLayers.add();
        newLayer.name = "Blemish Mask";
        doc.activeLayer = newLayer;
        return newLayer;
    }

    function importLayerFromFile(file, layerName, opacityVal) {
        if (!file || !file.exists || file.length < 300) {
            return null;
        }
        var doc = app.activeDocument;
        var origRuler = app.preferences.rulerUnits;
        try { app.preferences.rulerUnits = Units.PIXELS; } catch (e) {}

        var tempDoc = app.open(file);
        var importedLayer = tempDoc.activeLayer.duplicate(doc, ElementPlacement.PLACEATBEGINNING);
        tempDoc.close(SaveOptions.DONOTSAVECHANGES);

        try { app.preferences.rulerUnits = origRuler; } catch (e) {}

        doc.activeLayer = importedLayer;
        if (layerName) importedLayer.name = layerName;
        if (opacityVal !== undefined && opacityVal !== null) importedLayer.opacity = opacityVal;
        return importedLayer;
    }

    function placeFile(file) {
        return importLayerFromFile(file, null, 100);
    }

    function executeHeal(pad, textureBlend, featherRadius, grainIntensity, hideMask) {
        if (app.documents.length === 0) {
            alert("Please open a photo first.", "AI Retouch");
            return false;
        }

        var hCheck = checkBackendHealth();
        if (!hCheck.online) {
            alert("AI Retouch backend server is offline!\n\nPlease start the server by double-clicking:\nbackend\\run_server.bat\n\nOnce the server console is ready, retry.", "AI Server Offline");
            return false;
        }

        var doc = app.activeDocument;
        var maskLayer = doc.activeLayer;

        if (!maskLayer || maskLayer.isBackgroundLayer) {
            alert("Please select the painted mask layer (not Background).", "Select Mask Layer");
            return false;
        }

        var bounds = maskLayer.bounds;
        var left = bounds[0].as("px");
        var top = bounds[1].as("px");
        var right = bounds[2].as("px");
        var bottom = bounds[3].as("px");

        var width = right - left;
        var height = bottom - top;

        if (width <= 0 || height <= 0) {
            alert("The selected mask layer is empty.\nPlease paint over blemishes with a brush first.", "Mask Layer Empty");
            return false;
        }

        var docWidth = doc.width.as("px");
        var docHeight = doc.height.as("px");

        var cropLeft = Math.max(0, Math.floor(left - pad));
        var cropTop = Math.max(0, Math.floor(top - pad));
        var cropRight = Math.min(docWidth, Math.ceil(right + pad));
        var cropBottom = Math.min(docHeight, Math.ceil(bottom + pad));

        var cropW = cropRight - cropLeft;
        var cropH = cropBottom - cropTop;

        var cropBox = [
            [cropLeft, cropTop],
            [cropRight, cropTop],
            [cropRight, cropBottom],
            [cropLeft, cropBottom]
        ];

        var origActiveLayer = doc.activeLayer;
        var origMaskVisible = maskLayer.visible;

        maskLayer.visible = false;
        var imageCropFile = new File(tempFolder.fsName + "/image_crop.png");
        if (imageCropFile.exists) imageCropFile.remove();

        var dupImageDoc = doc.duplicate("AI_Temp_Image", false);
        try { dupImageDoc.selection.deselect(); } catch (e) {}
        try {
            if (dupImageDoc.bitsPerChannel !== BitsPerChannelType.EIGHT) {
                dupImageDoc.bitsPerChannel = BitsPerChannelType.EIGHT;
            }
        } catch (e) {}
        dupImageDoc.selection.select(cropBox);
        dupImageDoc.crop(dupImageDoc.selection.bounds);
        dupImageDoc.flatten();

        var pngSaveOptions = new PNGSaveOptions();
        pngSaveOptions.compression = 0;
        pngSaveOptions.interlaced = false;
        dupImageDoc.saveAs(imageCropFile, pngSaveOptions, true, Extension.LOWERCASE);
        dupImageDoc.close(SaveOptions.DONOTSAVECHANGES);

        maskLayer.visible = true;
        for (var k = 0; k < doc.artLayers.length; k++) {
            if (doc.artLayers[k] !== maskLayer) {
                doc.artLayers[k].visible = false;
            }
        }

        var maskCropFile = new File(tempFolder.fsName + "/mask_crop.png");
        if (maskCropFile.exists) maskCropFile.remove();

        var dupMaskDoc = doc.duplicate("AI_Temp_Mask", false);
        try { dupMaskDoc.selection.deselect(); } catch (e) {}
        try {
            if (dupMaskDoc.bitsPerChannel !== BitsPerChannelType.EIGHT) {
                dupMaskDoc.bitsPerChannel = BitsPerChannelType.EIGHT;
            }
        } catch (e) {}
        dupMaskDoc.selection.select(cropBox);
        dupMaskDoc.crop(dupMaskDoc.selection.bounds);
        dupMaskDoc.flatten();
        dupMaskDoc.saveAs(maskCropFile, pngSaveOptions, true, Extension.LOWERCASE);
        dupMaskDoc.close(SaveOptions.DONOTSAVECHANGES);

        for (var m = 0; m < doc.artLayers.length; m++) {
            doc.artLayers[m].visible = true;
        }
        if (hideMask) {
            maskLayer.visible = false;
        }

        var healedFile = new File(tempFolder.fsName + "/healed_patch.png");
        if (healedFile.exists) healedFile.remove();

        var imgPath = imageCropFile.fsName.replace(/\\/g, "/");
        var mskPath = maskCropFile.fsName.replace(/\\/g, "/");
        var outPath = healedFile.fsName.replace(/\\/g, "/");

        var cmd = 'curl.exe -s -S --tcp-nodelay -m 120 -X POST ' +
            '-F "image=@' + imgPath + '" ' +
            '-F "mask=@' + mskPath + '" ' +
            '-F "texture_blend=' + textureBlend + '" ' +
            '-F "feather_radius=' + featherRadius + '" ' +
            '-F "grain_intensity=' + grainIntensity + '" ' +
            '-o "' + outPath + '" ' +
            '"' + activeServerUrl + '/inpaint"';

        app.system(cmd);

        if (!healedFile.exists || healedFile.length < 500) {
            alert("Inpainting failed. Verify backend server is running on " + activeServerUrl, "AI Server Error");
            return false;
        }

        var placedLayer = importLayerFromFile(healedFile, "AI Healed Patch", 100);
        if (!placedLayer) return false;

        var placedBounds = placedLayer.bounds;
        var pLeft = placedBounds[0].as("px");
        var pTop = placedBounds[1].as("px");

        var dx = cropLeft - pLeft;
        var dy = cropTop - pTop;
        placedLayer.translate(dx, dy);

        return true;
    }

    // -------------------------------------------------------------
    // Modal Dialog Window (Dark Studio Design)
    // -------------------------------------------------------------
    var dlg = new Window("dialog", "AI Retouch - Fast Spot Remover");
    dlg.orientation = "column";
    dlg.alignChildren = ["fill", "top"];
    dlg.spacing = 10;
    dlg.margins = 14;

    var cBg = [0.14, 0.14, 0.14, 1.0];        // #242424
    var cPanel = [0.18, 0.18, 0.18, 1.0];     // #2E2E2E
    var cTextPrimary = [0.95, 0.95, 0.95, 1.0];
    var cTextMuted = [0.65, 0.65, 0.65, 1.0];
    var cGreen = [0.25, 0.78, 0.40, 1.0];
    var cRed = [0.85, 0.35, 0.35, 1.0];

    try {
        dlg.graphics.backgroundColor = dlg.graphics.newBrush(dlg.graphics.BrushType.SOLID_COLOR, cBg);
    } catch (e) {}

    // --- Header & Status ---
    var health = checkBackendHealth();
    var pnlHeader = dlg.add("group");
    pnlHeader.orientation = "row";
    pnlHeader.alignChildren = ["fill", "center"];

    var lblTitle = pnlHeader.add("statictext", undefined, "AI SPOT REMOVER");
    try {
        lblTitle.graphics.font = ScriptUI.newFont("Segoe UI", "BOLD", 12);
        lblTitle.graphics.foregroundColor = dlg.graphics.newPen(dlg.graphics.PenType.SOLID_COLOR, cTextPrimary, 1);
    } catch (e) {}

    var statusTextVal = health.online ? "[ONLINE: " + health.url.replace("http://127.0.0.1:", "PORT ") + "]" : "[OFFLINE: START SERVER]";
    var txtStatus = pnlHeader.add("statictext", undefined, statusTextVal);
    txtStatus.alignment = ["right", "center"];
    try {
        txtStatus.graphics.font = ScriptUI.newFont("Segoe UI", "BOLD", 10);
        txtStatus.graphics.foregroundColor = dlg.graphics.newPen(dlg.graphics.PenType.SOLID_COLOR, health.online ? cGreen : cRed, 1);
    } catch (e) {}

    // --- Step 1 Panel ---
    var pnlStep1 = dlg.add("panel", undefined, "Step 1: Paint Mask");
    pnlStep1.orientation = "column";
    pnlStep1.alignChildren = ["fill", "top"];
    pnlStep1.margins = 10;
    try { pnlStep1.graphics.backgroundColor = pnlStep1.graphics.newBrush(pnlStep1.graphics.BrushType.SOLID_COLOR, cPanel); } catch (e) {}

    var btnNewMask = pnlStep1.add("button", undefined, "Create New Mask Layer");
    btnNewMask.preferredSize.height = 28;
    btnNewMask.onClick = function () {
        createMaskLayer();
    };

    // --- Step 2 Panel ---
    var pnlStep2 = dlg.add("panel", undefined, "Step 2: AI Inpainting Parameters");
    pnlStep2.orientation = "column";
    pnlStep2.alignChildren = ["fill", "top"];
    pnlStep2.margins = 10;
    pnlStep2.spacing = 6;
    try { pnlStep2.graphics.backgroundColor = pnlStep2.graphics.newBrush(pnlStep2.graphics.BrushType.SOLID_COLOR, cPanel); } catch (e) {}

    var grpPad = pnlStep2.add("group");
    grpPad.orientation = "row";
    var lblPadT = grpPad.add("statictext", undefined, "Padding Area:");
    lblPadT.preferredSize.width = 110;
    var lblPad = grpPad.add("statictext", undefined, "20px");
    lblPad.preferredSize.width = 40;
    var sldPad = pnlStep2.add("slider", undefined, 20, 10, 60);
    sldPad.onChanging = function () { lblPad.text = Math.round(sldPad.value) + "px"; };

    var grpTex = pnlStep2.add("group");
    grpTex.orientation = "row";
    var lblTexT = grpTex.add("statictext", undefined, "Skin Texture:");
    lblTexT.preferredSize.width = 110;
    var lblTex = grpTex.add("statictext", undefined, "25%");
    lblTex.preferredSize.width = 40;
    var sldTex = pnlStep2.add("slider", undefined, 25, 0, 100);
    sldTex.onChanging = function () { lblTex.text = Math.round(sldTex.value) + "%"; };

    var grpFea = pnlStep2.add("group");
    grpFea.orientation = "row";
    var lblFeaT = grpFea.add("statictext", undefined, "Feather Radius:");
    lblFeaT.preferredSize.width = 110;
    var lblFea = grpFea.add("statictext", undefined, "3px");
    lblFea.preferredSize.width = 40;
    var sldFea = pnlStep2.add("slider", undefined, 3, 0, 10);
    sldFea.onChanging = function () { lblFea.text = Math.round(sldFea.value) + "px"; };

    var chkHide = pnlStep2.add("checkbox", undefined, "Hide mask layer after healing");
    chkHide.value = true;
    try { chkHide.graphics.foregroundColor = dlg.graphics.newPen(dlg.graphics.PenType.SOLID_COLOR, cTextPrimary, 1); } catch (e) {}

    var btnHeal = dlg.add("button", undefined, "EXECUTE AI SPOT INPAINTING");
    btnHeal.preferredSize.height = 34;
    btnHeal.onClick = function () {
        var pad = Math.round(sldPad.value);
        var textureBlend = (Math.round(sldTex.value) / 100.0).toFixed(2);
        var featherRadius = Math.round(sldFea.value);
        var grainIntensity = 0.04;
        var hideMask = chkHide.value;

        btnHeal.text = "Inpainting...";
        btnHeal.enabled = false;
        dlg.update();

        executeHeal(pad, textureBlend, featherRadius, grainIntensity, hideMask);

        dlg.close();
    };

    var grpBottom = dlg.add("group");
    grpBottom.orientation = "row";
    grpBottom.alignChildren = ["right", "center"];

    var btnCheck = grpBottom.add("button", undefined, "Check Server");
    btnCheck.preferredSize = [100, 24];
    btnCheck.onClick = function () {
        var h = checkBackendHealth();
        txtStatus.text = h.online ? "[ONLINE: " + h.url.replace("http://127.0.0.1:", "PORT ") + "]" : "[OFFLINE: START SERVER]";
        try { txtStatus.graphics.foregroundColor = dlg.graphics.newPen(dlg.graphics.PenType.SOLID_COLOR, h.online ? cGreen : cRed, 1); } catch (e) {}
    };

    var btnClose = grpBottom.add("button", undefined, "Close");
    btnClose.preferredSize = [80, 24];
    btnClose.onClick = function () { dlg.close(); };

    dlg.center();
    dlg.show();
})();
