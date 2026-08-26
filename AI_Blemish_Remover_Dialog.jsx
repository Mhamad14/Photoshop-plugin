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

    function executeHeal(pad, textureBlend, featherRadius, grainIntensity, hideMask) {
        if (app.documents.length === 0) {
            alert("Please open a photo first.", "AI Retouch");
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

        var cmd = 'curl.exe -s -X POST ' +
            '-F "image=@' + imageCropFile.fsName + '" ' +
            '-F "mask=@' + maskCropFile.fsName + '" ' +
            '-F "texture_blend=' + textureBlend + '" ' +
            '-F "feather_radius=' + featherRadius + '" ' +
            '-F "grain_intensity=' + grainIntensity + '" ' +
            '-o "' + healedFile.fsName + '" ' +
            '"' + activeServerUrl + '/inpaint"';

        app.system(cmd);

        if (!healedFile.exists || healedFile.length < 500) {
            alert("Inpainting failed. Verify backend server is running on " + activeServerUrl, "AI Server Error");
            return false;
        }

        placeFile(healedFile);

        var placedLayer = doc.activeLayer;
        placedLayer.name = "AI Healed Patch";

        var placedBounds = placedLayer.bounds;
        var pLeft = placedBounds[0].as("px");
        var pTop = placedBounds[1].as("px");

        var dx = cropLeft - pLeft;
        var dy = cropTop - pTop;
        placedLayer.translate(dx, dy);

        return true;
    }

    // Modal Dialog Window
    var dlg = new Window("dialog", "AI Retouch - Fast Spot Remover");
    dlg.orientation = "column";
    dlg.alignChildren = ["fill", "top"];
    dlg.spacing = 10;
    dlg.margins = 16;

    // --- Status Panel ---
    var health = checkBackendHealth();
    var pnlStatus = dlg.add("panel", undefined, "Backend Status");
    pnlStatus.orientation = "row";
    pnlStatus.alignChildren = ["left", "center"];
    var statusTextVal = health.online ? "[Online] " + health.url.replace("http://127.0.0.1:", "Port ") : "[Offline] Start backend server";
    var txtStatus = pnlStatus.add("statictext", undefined, statusTextVal);
    txtStatus.preferredSize.width = 220;
    var btnCheck = pnlStatus.add("button", undefined, "Check");
    btnCheck.onClick = function () {
        var h = checkBackendHealth();
        txtStatus.text = h.online ? "[Online] " + h.url.replace("http://127.0.0.1:", "Port ") : "[Offline] Start backend server";
    };

    // --- Step 1 Panel ---
    var pnlStep1 = dlg.add("panel", undefined, "Step 1: Paint Mask");
    pnlStep1.orientation = "column";
    pnlStep1.alignChildren = ["fill", "top"];
    var btnNewMask = pnlStep1.add("button", undefined, "Create New Mask Layer");
    btnNewMask.preferredSize.height = 30;
    btnNewMask.onClick = function () {
        createMaskLayer();
    };

    // --- Step 2 Panel ---
    var pnlStep2 = dlg.add("panel", undefined, "Step 2: AI Inpainting Parameters");
    pnlStep2.orientation = "column";
    pnlStep2.alignChildren = ["fill", "top"];
    pnlStep2.spacing = 6;

    var grpPad = pnlStep2.add("group");
    grpPad.orientation = "row";
    grpPad.add("statictext", undefined, "Padding:");
    var lblPad = grpPad.add("statictext", undefined, "20px");
    lblPad.preferredSize.width = 40;
    var sldPad = pnlStep2.add("slider", undefined, 20, 10, 60);
    sldPad.onChanging = function () { lblPad.text = Math.round(sldPad.value) + "px"; };

    var grpTex = pnlStep2.add("group");
    grpTex.orientation = "row";
    grpTex.add("statictext", undefined, "Skin Texture:");
    var lblTex = grpTex.add("statictext", undefined, "25%");
    lblTex.preferredSize.width = 40;
    var sldTex = pnlStep2.add("slider", undefined, 25, 0, 100);
    sldTex.onChanging = function () { lblTex.text = Math.round(sldTex.value) + "%"; };

    var grpFea = pnlStep2.add("group");
    grpFea.orientation = "row";
    grpFea.add("statictext", undefined, "Feather:");
    var lblFea = grpFea.add("statictext", undefined, "3px");
    lblFea.preferredSize.width = 40;
    var sldFea = pnlStep2.add("slider", undefined, 3, 0, 10);
    sldFea.onChanging = function () { lblFea.text = Math.round(sldFea.value) + "px"; };

    var chkHide = pnlStep2.add("checkbox", undefined, "Hide mask layer after healing");
    chkHide.value = true;

    var btnHeal = dlg.add("button", undefined, "Heal Painted Blemishes");
    btnHeal.preferredSize.height = 36;
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

    var btnClose = dlg.add("button", undefined, "Cancel");
    btnClose.onClick = function () { dlg.close(); };

    dlg.center();
    dlg.show();
})();
