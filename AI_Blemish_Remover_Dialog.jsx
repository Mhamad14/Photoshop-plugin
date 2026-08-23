#target photoshop

/**
 * AI Retouch - Blemish & Pimple Remover (Modal Dialog Version)
 * For quick 1-click inpainting in Photoshop 2020+
 */

(function () {
    var SERVER_CANDIDATES = [
        "http://127.0.0.1:8008",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8001"
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
            alert("Please open an image in Photoshop first!", "AI Retouch");
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
            alert("Please open a photo first!", "AI Retouch");
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
            alert("The selected mask layer is empty!\nPlease paint over blemishes with a brush first.", "Mask Layer Empty");
            return false;
        }

        var docWidth = doc.width.as("px");
        var docHeight = doc.height.as("px");

        var cropLeft = Math.max(0, Math.floor(left - pad));
        var cropTop = Math.max(0, Math.floor(top - pad));
        var cropRight = Math.min(docWidth, Math.ceil(right + pad));
        var cropBottom = Math.min(docHeight, Math.ceil(bottom + pad));

        var maskLayerName = maskLayer.name;
        var baseCropFile = new File(tempFolder.fsName + "/base_crop.png");
        var maskCropFile = new File(tempFolder.fsName + "/mask_crop.png");
        var healedPatchFile = new File(tempFolder.fsName + "/healed_patch.png");

        if (baseCropFile.exists) baseCropFile.remove();
        if (maskCropFile.exists) maskCropFile.remove();
        if (healedPatchFile.exists) healedPatchFile.remove();

        var pngSaveOptions = new PNGSaveOptions();
        pngSaveOptions.compression = 0;
        pngSaveOptions.interlaced = false;

        // 1. Export Base Crop
        var baseDoc = doc.duplicate("AI_Base_Crop_Temp", false);
        try {
            for (var i = 0; i < baseDoc.layers.length; i++) {
                if (baseDoc.layers[i].name === maskLayerName) {
                    baseDoc.layers[i].visible = false;
                }
            }
        } catch (e) {}

        baseDoc.flatten();
        baseDoc.crop([
            UnitValue(cropLeft, "px"),
            UnitValue(cropTop, "px"),
            UnitValue(cropRight, "px"),
            UnitValue(cropBottom, "px")
        ]);
        baseDoc.saveAs(baseCropFile, pngSaveOptions, true, Extension.LOWERCASE);
        baseDoc.close(SaveOptions.DONOTSAVECHANGES);

        // 2. Export Mask Crop
        var maskDoc = doc.duplicate("AI_Mask_Crop_Temp", false);
        for (var j = 0; j < maskDoc.layers.length; j++) {
            if (maskDoc.layers[j].name === maskLayerName) {
                maskDoc.layers[j].visible = true;
            } else {
                maskDoc.layers[j].visible = false;
            }
        }
        maskDoc.crop([
            UnitValue(cropLeft, "px"),
            UnitValue(cropTop, "px"),
            UnitValue(cropRight, "px"),
            UnitValue(cropBottom, "px")
        ]);
        maskDoc.saveAs(maskCropFile, pngSaveOptions, true, Extension.LOWERCASE);
        maskDoc.close(SaveOptions.DONOTSAVECHANGES);

        // 3. Call FastAPI Backend
        var cmd = 'curl.exe -s -X POST ' +
            '-F "image=@' + baseCropFile.fsName + '" ' +
            '-F "mask=@' + maskCropFile.fsName + '" ' +
            '-F "texture_blend=' + textureBlend + '" ' +
            '-F "feather_radius=' + featherRadius + '" ' +
            '-F "grain_intensity=' + grainIntensity + '" ' +
            '-o "' + healedPatchFile.fsName + '" ' +
            '"' + activeServerUrl + '/heal-blemish"';

        app.system(cmd);

        if (!healedPatchFile.exists || healedPatchFile.length < 100) {
            alert("AI inpainting request failed.\n\nMake sure 'run_server.bat' is running at " + activeServerUrl, "AI Server Error");
            return false;
        }

        // 4. Place and offset patch
        placeFile(healedPatchFile);

        var placedLayer = doc.activeLayer;
        placedLayer.name = "AI Blemish Removal";

        var placedBounds = placedLayer.bounds;
        var curLeft = placedBounds[0].as("px");
        var curTop = placedBounds[1].as("px");

        var deltaX = cropLeft - curLeft;
        var deltaY = cropTop - curTop;

        if (deltaX !== 0 || deltaY !== 0) {
            placedLayer.translate(UnitValue(deltaX, "px"), UnitValue(deltaY, "px"));
        }

        if (hideMask) {
            maskLayer.visible = false;
        }

        return true;
    }

    // Modal Dialog Window
    var dlg = new Window("dialog", "AI Blemish Remover", undefined);
    dlg.orientation = "column";
    dlg.alignChildren = ["fill", "top"];
    dlg.spacing = 8;
    dlg.margins = 14;

    // Server Status
    var health = checkBackendHealth();
    var pnlStatus = dlg.add("panel", undefined, "AI Server Status");
    pnlStatus.orientation = "row";
    var txtStatus = pnlStatus.add("statictext", undefined, health.online ? "● Online (" + health.url.replace("http://127.0.0.1:", "Port ") + ")" : "● Offline (Start run_server.bat)");
    txtStatus.preferredSize.width = 200;

    // Actions
    var btnMask = dlg.add("button", undefined, "1. Create 'Blemish Mask' Layer");
    btnMask.preferredSize.height = 30;
    btnMask.onClick = function () {
        createMaskLayer();
        dlg.close();
    };

    var btnHeal = dlg.add("button", undefined, "2. Heal Painted Blemishes");
    btnHeal.preferredSize.height = 36;

    // Settings
    var pnlSettings = dlg.add("panel", undefined, "Settings");
    pnlSettings.orientation = "column";
    pnlSettings.alignChildren = ["fill", "top"];
    pnlSettings.spacing = 6;

    var grpPad = pnlSettings.add("group");
    grpPad.orientation = "row";
    grpPad.add("statictext", undefined, "Padding:");
    var lblPad = grpPad.add("statictext", undefined, "20px");
    lblPad.preferredSize.width = 40;
    var sldPad = pnlSettings.add("slider", undefined, 20, 10, 60);
    sldPad.onChanging = function () { lblPad.text = Math.round(sldPad.value) + "px"; };

    var grpTex = pnlSettings.add("group");
    grpTex.orientation = "row";
    grpTex.add("statictext", undefined, "Skin Texture:");
    var lblTex = grpTex.add("statictext", undefined, "25%");
    lblTex.preferredSize.width = 40;
    var sldTex = pnlSettings.add("slider", undefined, 25, 0, 100);
    sldTex.onChanging = function () { lblTex.text = Math.round(sldTex.value) + "%"; };

    var grpFea = pnlSettings.add("group");
    grpFea.orientation = "row";
    grpFea.add("statictext", undefined, "Feather:");
    var lblFea = grpFea.add("statictext", undefined, "3px");
    lblFea.preferredSize.width = 40;
    var sldFea = pnlSettings.add("slider", undefined, 3, 0, 10);
    sldFea.onChanging = function () { lblFea.text = Math.round(sldFea.value) + "px"; };

    var chkHide = pnlSettings.add("checkbox", undefined, "Hide mask layer after healing");
    chkHide.value = true;

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
