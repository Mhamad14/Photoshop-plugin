# AI Retouching MVP — Photoshop UXP + FastAPI + LaMa Inpainting

An ultra-fast, non-destructive blemish and pimple removal plugin for Adobe Photoshop.

Powered by a local **FastAPI** Python server running **Simple-LaMa AI inpainting**, seamlessly connected to a modern **Photoshop UXP** panel.

---

## ⚡ Key Highlights & Architecture

- **Crop-to-Bounding-Box Pipeline**: Instead of sending the full multi-megapixel portrait over the wire, the UXP plugin calculates the bounding box of the painted blemish mask (+20px padding) and sends only that small crop. This keeps inference times sub-second and memory usage minimal.
- **100% Non-Destructive**: The AI server returns only the healed patch as a transparent PNG (alpha = feathered mask). The UXP plugin places this patch onto a new isolated layer positioned at the exact pixel offset.
- **Skin Realism & Texture Preservation**: Skin inpainting models can sometimes produce plastic/smooth skin. The backend includes high-frequency skin texture retention and micro-grain matching.
- **Local & Private**: All inference runs 100% locally on your machine with CUDA/GPU acceleration (or fast CPU fallback). No external cloud calls or subscriptions.

---

## 📁 Project Structure

```
Photoshop-plugin/
├── backend/
│   ├── server.py              # FastAPI server with Simple-LaMa & texture blending
│   ├── requirements.txt       # Python dependencies
│   ├── test_client.py         # End-to-end backend testing script
│   ├── run_server.py          # Python server launcher with dependency verification
│   └── run_server.bat         # One-click Windows batch launcher
├── uxp-plugin/
│   ├── manifest.json          # UXP Manifest v5 specification
│   ├── index.html             # Photoshop panel UI layout
│   ├── index.css              # Dark theme styling matching Photoshop
│   ├── index.js               # UXP batchPlay bridge, cropping, & positioning logic
│   └── icons/                 # Plugin icon assets (23x23 & 48x48)
└── README.md                  # Complete documentation and setup guide
```

---

## 🚀 Quick Start Guide

### Step 1: Start the Local AI Server

1. Open a terminal in `backend/` or double-click `backend/run_server.bat`.
2. Or run manually:
   ```bash
   pip install -r backend/requirements.txt
   python backend/server.py
   ```
3. The server will start on `http://127.0.0.1:8765` (falls back to 8766/8001 if busy — 8000 is avoided since PHP dev servers commonly occupy it). On first run, it will automatically download the lightweight LaMa model weights (~200MB).

To verify the server is running correctly:
```bash
python backend/test_client.py
```

---

### Optional: Build a Better Local Detector

The panel includes an opt-in **Training Mode** under the AI refinement section. After analyzing a portrait, enable Training Mode, choose a click label, review every detection, and confirm that you have permission to use the portrait for training. Click **Save Reviewed Sample** to write the portrait, YOLO segmentation labels, and review metadata to `backend/training/data/retouch_skin/`.

Training Mode does not train automatically and does not upload images. It only saves samples after explicit review and permission confirmation. See `backend/training/README.md` for annotation rules and the YOLOv8-seg training command.

---

### Step 2: Load the UXP Plugin into Photoshop

1. Open the **Adobe UXP Developer Tool** (available via Adobe Creative Cloud desktop app).
2. Click **Add Plugin** and select `uxp-plugin/manifest.json`.
3. Open **Adobe Photoshop** (version 23.0 / 2022 or newer).
4. In the UXP Developer Tool, click **Actions (•••)** next to the plugin and select **Load**.
5. The **AI Blemish Remover** panel will appear in Photoshop under `Plugins > AI Retouch > AI Blemish Remover`.

---

## 🎨 How to Use in Photoshop

1. **Open your Portrait** in Photoshop.
2. In the AI Retouch panel, click **Create "Blemish Mask" Layer** (or create a transparent layer manually).
3. Using any standard hard or soft brush (with any color, e.g. white or red), **paint over the pimples, blemishes, or spots** you want to remove.
4. Click **⚡ Heal Painted Blemishes**.
5. The plugin will:
   - Calculate the bounding box of your brush strokes.
   - Crop the base portrait and mask.
   - Send them to the local Simple-LaMa engine.
   - Place the healed result on a new layer named `AI Blemish Removal` positioned at the exact pixel coordinates.
   - The original image remains completely untouched!

---

## 🎛 Fine-Tuning Controls

- **Crop Padding (px)**: Margins added around the painted blemish mask (default: 20px). Gives the AI surrounding context to sample natural skin tone.
- **Skin Texture Blend (%)**: Re-injects high-frequency skin texture and pores from the original image (default: 25%).
- **Edge Feather (px)**: Softens the alpha boundary of the healed patch for seamless blending into the base photo (default: 3px).
- **Micro-Grain Overlay (%)**: Synthesizes matched subtle camera sensor grain to prevent plastic look (default: 4%).
- **Hide mask layer after healing**: Automatically toggles off the visibility of the brush mask layer once healed.

---

## 🛠 API Reference

### `GET /health`
Returns the status of the server, compute device (`cuda` or `cpu`), and model readiness.

### `POST /heal-blemish`
- **Form Data**:
  - `image`: Base image crop (`.png` / `.jpg`)
  - `mask`: Painted mask crop (`.png`)
  - `texture_blend`: Float `0.0` to `1.0` (default `0.25`)
  - `feather_radius`: Integer pixels `0` to `10` (default `3`)
  - `grain_intensity`: Float `0.0` to `0.2` (default `0.04`)
- **Returns**: Transparent PNG patch where alpha is the feathered mask.
