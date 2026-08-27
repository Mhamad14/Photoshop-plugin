import base64
import io
import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageFilter
import torch

from blemish_detector import auto_detect_blemishes
from gemini_detector import detect_blemishes_gemini
from face_segmenter import segment_face_skin, get_bisenet_parser
from pimple_detector_v2 import detect_pimple_candidates, blobs_to_mask, add_blob_at_point, toggle_blob_at_point
from skin_toner import calculate_tone_lift, create_lightened_rgba_patch
from skin_smoother import apply_full_smooth, create_smooth_rgba_patch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ai_retouch_server")

lama_model = None
device_name = "cpu"

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(data: dict):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save config: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global lama_model, device_name
    logger.info("Initializing SimpleLama Inpainting & BiSeNet models...")
    start_time = time.time()
    
    if torch.cuda.is_available():
        device_name = f"cuda ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    
    logger.info(f"Compute device selected: {device_name}")
    
    # Warm up BiSeNet Face Parser
    try:
        get_bisenet_parser()
    except Exception as e:
        logger.warning(f"Pre-warming BiSeNet parser: {e}")

    try:
        from simple_lama_inpainting import SimpleLama
        lama_model = SimpleLama()
        load_time = time.time() - start_time
        logger.info(f"SimpleLama model loaded successfully in {load_time:.2f}s on {device_name}")
    except Exception as e:
        logger.error(f"Failed to load SimpleLama model: {e}", exc_info=True)
        lama_model = None
    
    yield
    
    lama_model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Server shutting down, resources released.")

app = FastAPI(
    title="Photoshop AI Retouching Backend - v2 Auto-Detection",
    description="Auto-Detection Architecture: BiSeNet Skin Segmentation, Hybrid Pimple Detection, Live Refinement & LaMa Inpainting.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    cfg = load_config()
    has_gemini = bool(cfg.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY"))
    return {
        "status": "ready" if lama_model is not None else "model_not_loaded",
        "model": "simple-lama-inpainting + bisenet-face-parsing",
        "version": "2.0.0",
        "device": device_name,
        "cuda_available": torch.cuda.is_available(),
        "gemini_enabled": has_gemini
    }


@app.post("/set-api-key")
async def set_api_key(gemini_api_key: str = Form(...)):
    key = gemini_api_key.strip()
    cfg = load_config()
    cfg["gemini_api_key"] = key
    save_config(cfg)
    os.environ["GEMINI_API_KEY"] = key
    return {"status": "saved", "gemini_enabled": bool(key)}


def neutralize_erythema(img_rgb: np.ndarray, mask_gray: np.ndarray) -> np.ndarray:
    """
    Dermatological Pre-processing:
    Neutralizes inflamed red/brown erythema cast and dark/whitehead core under acne blemishes
    using smooth guided background propagation from surrounding healthy skin.
    """
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_c, a_c, b_c = cv2.split(img_lab)

    # Surrounding healthy skin weight
    healthy_weight = (mask_gray == 0).astype(np.float32)
    ksize = max(21, int(max(img_rgb.shape[:2]) * 0.03)) | 1

    blurred_l = cv2.GaussianBlur(l_c * healthy_weight, (ksize, ksize), 0)
    blurred_a = cv2.GaussianBlur(a_c * healthy_weight, (ksize, ksize), 0)
    blurred_b = cv2.GaussianBlur(b_c * healthy_weight, (ksize, ksize), 0)
    norm_weight = cv2.GaussianBlur(healthy_weight, (ksize, ksize), 0) + 1e-5

    healthy_l = blurred_l / norm_weight
    healthy_a = blurred_a / norm_weight
    healthy_b = blurred_b / norm_weight

    mask_factor = (mask_gray.astype(np.float32) / 255.0)
    
    # Correct redness (a*), pigment/pus (b*), and tone/luminance crater/apex (L*)
    a_corrected = a_c * (1.0 - mask_factor * 0.85) + healthy_a * (mask_factor * 0.85)
    b_corrected = b_c * (1.0 - mask_factor * 0.70) + healthy_b * (mask_factor * 0.70)
    l_corrected = l_c * (1.0 - mask_factor * 0.60) + healthy_l * (mask_factor * 0.60)

    lab_clean = cv2.merge([l_corrected, a_corrected, b_corrected]).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_clean, cv2.COLOR_LAB2RGB)


def blend_skin_texture(
    original_img: np.ndarray,
    inpainted_img: np.ndarray,
    mask_gray: np.ndarray,
    texture_blend: float = 0.25,
    grain_intensity: float = 0.03
) -> np.ndarray:
    """
    Pore-Preserving Texture Transfer & Illumination Alignment:
    1. Samples high-frequency spatial skin pore texture from healthy surrounding skin annulus (NOT from the unhealed blemish).
    2. Aligns local illumination gradient to prevent lighter/darker ghost circles.
    3. Adds subtle micro-grain matched to authentic skin texture.
    """
    h, w, c = inpainted_img.shape
    orig_f = original_img.astype(np.float32)
    inpaint_f = inpainted_img.astype(np.float32)
    mask_f = (mask_gray.astype(np.float32) / 255.0)[:, :, None]

    if texture_blend > 0:
        # Extract high-frequency pores from the image outside the mask
        blurred_orig = cv2.GaussianBlur(orig_f, (5, 5), 0)
        high_freq_all = orig_f - blurred_orig

        # Zero out the high frequencies inside the blemish mask so pimple roughness/scabs are excluded
        healthy_mask = (mask_gray == 0).astype(np.float32)[:, :, None]
        high_freq_healthy = high_freq_all * healthy_mask

        # Propagate healthy pore texture across the blemish spots using normalized blur
        k_tex = max(11, int(min(h, w) * 0.02)) | 1
        prop_tex = cv2.GaussianBlur(high_freq_healthy, (k_tex, k_tex), 0)
        norm_tex = cv2.GaussianBlur(healthy_mask.squeeze(-1), (k_tex, k_tex), 0)[:, :, None] + 1e-5
        synthesized_pores = prop_tex / norm_tex

        # Re-inject realistic healthy pores into the inpainted spot
        inpaint_f = inpaint_f + (synthesized_pores * (texture_blend * 1.2) * mask_f)

    if grain_intensity > 0:
        # Add micro-grain matched to skin tone
        noise = np.random.normal(loc=0.0, scale=grain_intensity * 255.0, size=(h, w, c))
        inpaint_f = inpaint_f + (noise * mask_f)

    # Local Illumination Gradient Alignment (Prevents ghost circles on shadows/highlights)
    # Match low-frequency illumination of inpainted patch to healthy surroundings
    k_illum = max(31, int(min(h, w) * 0.05)) | 1
    orig_illum = cv2.GaussianBlur(orig_f, (k_illum, k_illum), 0)
    inpaint_illum = cv2.GaussianBlur(inpaint_f, (k_illum, k_illum), 0)
    illum_delta = (orig_illum - inpaint_illum) * mask_f * 0.45
    inpaint_f = inpaint_f + illum_delta

    return np.clip(inpaint_f, 0, 255).astype(np.uint8)


def inpaint_with_context_tiling(
    model,
    img_rgb: np.ndarray,
    mask_gray: np.ndarray,
    max_tile_size: int = 768,
    context_pad: int = 80
) -> np.ndarray:
    """
    Context-Aware Tiled Inpainting:
    Runs Simple-LaMa at high resolution for blemish clusters with context margins,
    preserving 100% native portrait sharpness without whole-canvas downscaling blur.
    """
    h, w, _ = img_rgb.shape
    if model is None:
        return img_rgb.copy()

    # If entire image is reasonably sized, run direct inpainting
    if max(h, w) <= max_tile_size:
        pil_in = model(Image.fromarray(img_rgb), Image.fromarray(mask_gray))
        if pil_in.size != (w, h):
            pil_in = pil_in.resize((w, h), Image.Resampling.BILINEAR)
        return np.array(pil_in)

    # Find connected components of blemish mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask_gray > 10).astype(np.uint8))
    if num_labels <= 1:
        return img_rgb.copy()

    output_rgb = img_rgb.copy()

    # Cluster nearby blemishes together
    dilated_clusters = cv2.dilate((mask_gray > 10).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (context_pad, context_pad)))
    c_num, c_labels, c_stats, _ = cv2.connectedComponentsWithStats(dilated_clusters)

    for c in range(1, c_num):
        bx = int(c_stats[c, cv2.CC_STAT_LEFT])
        by = int(c_stats[c, cv2.CC_STAT_TOP])
        bw = int(c_stats[c, cv2.CC_STAT_WIDTH])
        bh = int(c_stats[c, cv2.CC_STAT_HEIGHT])

        # Add context padding
        x1 = max(0, bx - context_pad)
        y1 = max(0, by - context_pad)
        x2 = min(w, bx + bw + context_pad)
        y2 = min(h, by + bh + context_pad)

        crop_w = x2 - x1
        crop_h = y2 - y1

        crop_img = img_rgb[y1:y2, x1:x2]
        crop_mask = mask_gray[y1:y2, x1:x2]

        if np.sum(crop_mask > 0) == 0:
            continue

        # Inpaint localized crop
        try:
            crop_pil = model(Image.fromarray(crop_img), Image.fromarray(crop_mask))
            if crop_pil.size != (crop_w, crop_h):
                crop_pil = crop_pil.resize((crop_w, crop_h), Image.Resampling.BILINEAR)
            crop_inpainted = np.array(crop_pil)
        except Exception as e:
            logger.warning(f"Tile inpainting exception on crop [{x1}:{x2}, {y1}:{y2}]: {e}")
            continue

        # Feathered blend into output
        crop_alpha = cv2.GaussianBlur((crop_mask > 0).astype(np.float32), (7, 7), 0)[:, :, None]
        output_rgb[y1:y2, x1:x2] = np.clip(
            crop_inpainted.astype(np.float32) * crop_alpha + output_rgb[y1:y2, x1:x2].astype(np.float32) * (1.0 - crop_alpha),
            0, 255
        ).astype(np.uint8)

    return output_rgb


def apply_feather(mask_gray: np.ndarray, feather_radius: int = 3) -> np.ndarray:
    if feather_radius <= 0:
        return mask_gray
    ksize = feather_radius * 2 + 1
    return cv2.GaussianBlur(mask_gray, (ksize, ksize), 0)


# =====================================================================
# v2 AUTO-DETECTION ARCHITECTURE ENDPOINTS
# =====================================================================

@app.post("/analyze")
async def analyze_portrait(
    image: UploadFile = File(..., description="Full portrait image for auto-analysis"),
    sensitivity: float = Form(0.5, description="Blemish detection sensitivity (0.1 to 1.0)"),
    detect_pimples: bool = Form(True, description="Whether to detect blemishes"),
    detect_skin: bool = Form(True, description="Whether to segment skin"),
    include_neck: bool = Form(True, description="Include neck in skin segmentation"),
    feather_radius: int = Form(3, description="Edge feather radius for skin mask"),
    gemini_api_key: Optional[str] = Form(None, description="Optional Gemini API key")
):
    """
    Core v2 Auto-Detection Pipeline:
    1. Layer 1: BiSeNet face/skin segmentation -> skin_mask & sampled natural skin tone
    2. Layer 2: Classical CV + VLM blemish detector -> discrete pimple blobs with metadata
    Returns JSON with skin_mask base64, blob coordinates, and base tone.
    """
    t_start = time.time()
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    img_rgb = np.array(pil_image)
    h, w, _ = img_rgb.shape

    cfg = load_config()
    api_key = (gemini_api_key or "").strip() or cfg.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")

    # 1. Layer 1: Skin Segmentation
    skin_mask, skin_meta = segment_face_skin(
        img_rgb=img_rgb,
        include_neck=include_neck,
        feather_radius=feather_radius
    )

    # 2. Layer 2: Pimple Detection (within skin_mask)
    blobs = []
    if detect_pimples:
        blobs, _ = detect_pimple_candidates(
            img_rgb=img_rgb,
            skin_mask=skin_mask,
            sensitivity=max(0.05, min(1.0, sensitivity)),
            gemini_api_key=api_key
        )

    # Encode skin mask to PNG base64 for fast HTML canvas preview
    skin_pil = Image.fromarray(skin_mask)
    skin_buf = io.BytesIO()
    skin_pil.save(skin_buf, format="PNG", optimize=True)
    skin_b64 = "data:image/png;base64," + base64.b64encode(skin_buf.getvalue()).decode("ascii")

    process_time = round((time.time() - t_start) * 1000, 1)
    logger.info(f"Analyze completed in {process_time}ms: {len(blobs)} pimple blobs, {skin_meta['skin_percentage']:.1f}% skin coverage.")

    return JSONResponse({
        "status": "success",
        "image_size": [w, h],
        "skin_mask_base64": skin_b64,
        "skin_percentage": round(skin_meta["skin_percentage"], 1),
        "skin_pixel_count": skin_meta["skin_pixel_count"],
        "base_tone_rgb": skin_meta["base_tone_rgb"],
        "base_tone_lab": [round(x, 1) for x in skin_meta["base_tone_lab"]],
        "blobs": blobs,
        "blobs_count": len(blobs),
        "process_time_ms": process_time
    })


@app.post("/preview")
async def preview_result(
    image: UploadFile = File(..., description="Full portrait image"),
    blobs_json: Optional[str] = Form("[]", description="JSON list of all blemish blobs (active flags respected)"),
    skin_mask: Optional[UploadFile] = File(None, description="Cached skin mask PNG from /analyze (skips re-segmentation)"),
    include_heal: bool = Form(True, description="Include pimple removal in preview"),
    include_smooth: bool = Form(True, description="Include skin smoothing (texture + redness) in preview"),
    include_lighten: bool = Form(True, description="Include skin lightening in preview"),
    heal_mode: str = Form("full_inpaint", description="'full_inpaint' | 'calm_redness' | 'flatten_bump'"),
    smooth_strength: float = Form(0.45, description="Smoothing strength 0-1"),
    texture_keep: float = Form(0.4, description="Pore/texture retention during smoothing 0-1"),
    strength: float = Form(0.35, description="Lightening strength 0-1"),
    texture_blend: float = Form(0.25, description="Texture retention 0-1"),
    feather_radius: int = Form(3, description="Edge feather radius px"),
    grain_intensity: float = Form(0.03, description="Micro-grain intensity"),
    base_tone_lab: Optional[str] = Form(None, description="JSON [L,a,b] sampled base tone"),
    max_size: int = Form(720, ge=256, le=1600, description="Max long edge for fast live preview")
):
    """
    Layer 3 - Live Result Preview.
    Renders the FINAL retouched look (healed pimples + lightened skin) on a downscaled
    copy so the panel can show the user exactly how the result will look BEFORE applying.
    Fast enough to run on every refinement edit.
    """
    global lama_model
    t_start = time.time()
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    orig_w, orig_h = pil_image.size

    # Downscale for fast inference (preview only; Apply uses full resolution)
    scale = min(1.0, float(max_size) / float(max(orig_w, orig_h)))
    if scale < 1.0:
        prev_pil = pil_image.resize(
            (max(1, int(orig_w * scale)), max(1, int(orig_h * scale))),
            Image.Resampling.LANCZOS
        )
    else:
        prev_pil = pil_image.copy()

    img_rgb = np.array(prev_pil)
    h, w, _ = img_rgb.shape

    parsed_lab = None
    if base_tone_lab:
        try:
            parsed_lab = json.loads(base_tone_lab)
        except Exception:
            parsed_lab = None

    # Cached skin mask (preferred) or fresh segmentation fallback
    mask_np = None
    if include_lighten or include_smooth:
        if skin_mask is not None:
            try:
                mask_bytes = await skin_mask.read()
                mask_pil = Image.open(io.BytesIO(mask_bytes))
                mask_np = np.array(mask_pil.split()[3]) if mask_pil.mode == "RGBA" else np.array(mask_pil.convert("L"))
                if mask_np.shape[:2] != (h, w):
                    mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                logger.warning(f"Failed to read cached skin mask, re-segmenting: {e}")
                mask_np = None
        if mask_np is None:
            mask_np, _ = segment_face_skin(img_rgb, feather_radius=feather_radius)

    current_rgb = img_rgb.copy()
    healed_pixels = 0

    # --- Action 1: heal active blemish blobs ---
    if include_heal and blobs_json:
        try:
            all_blobs = json.loads(blobs_json)
        except Exception:
            all_blobs = []
        active_blobs = [b for b in all_blobs if b.get("active", True)]

        if active_blobs:
            # Blobs arrive in original-image coordinates -> rescale to preview space
            coord_scale = w / float(max(1, orig_w))
            scaled_blobs = []
            for b in active_blobs:
                bc = dict(b)
                bc["centroid"] = [b["centroid"][0] * coord_scale, b["centroid"][1] * coord_scale]
                bc["radius"] = max(2.0, b.get("radius", 6) * coord_scale)
                scaled_blobs.append(bc)

            pimple_mask = blobs_to_mask(scaled_blobs, (h, w), soft_falloff=False)
            healed_pixels = int(np.sum(pimple_mask > 0))

            if healed_pixels > 0:
                clean_rgb = neutralize_erythema(current_rgb, pimple_mask)

                if heal_mode == "calm_redness":
                    final_rgb = clean_rgb
                elif heal_mode == "flatten_bump":
                    orig_f = current_rgb.astype(np.float32)
                    blurred_low = cv2.GaussianBlur(orig_f, (15, 15), 0)
                    high_freq = orig_f - blurred_low
                    clean_f = clean_rgb.astype(np.float32)
                    mask_f = (pimple_mask.astype(np.float32) / 255.0)[:, :, None]
                    final_rgb = np.clip(clean_f + high_freq * (1.0 - mask_f * 0.7), 0, 255).astype(np.uint8)
                else:
                    if lama_model is None:
                        raise HTTPException(status_code=503, detail="AI inpainting model is not loaded.")
                    inpainted_np = inpaint_with_context_tiling(
                        model=lama_model,
                        img_rgb=clean_rgb,
                        mask_gray=pimple_mask,
                        max_tile_size=768,
                        context_pad=40
                    )
                    final_rgb = blend_skin_texture(
                        original_img=current_rgb,
                        inpainted_img=inpainted_np,
                        mask_gray=pimple_mask,
                        texture_blend=max(0.0, min(1.0, texture_blend)),
                        grain_intensity=max(0.0, min(0.2, grain_intensity))
                    )

                feathered_alpha = apply_feather(pimple_mask, feather_radius=max(0, feather_radius))
                alpha_f = (feathered_alpha.astype(np.float32) / 255.0)[:, :, None]
                current_rgb = np.clip(
                    final_rgb.astype(np.float32) * alpha_f + current_rgb.astype(np.float32) * (1.0 - alpha_f),
                    0, 255
                ).astype(np.uint8)

    # --- Action 2: smooth skin (redness evening + frequency separation) ---
    if include_smooth and mask_np is not None:
        smoothed_rgb, _ = apply_full_smooth(
            img_rgb=current_rgb,
            skin_mask=mask_np,
            strength=max(0.0, min(1.0, smooth_strength)),
            texture_keep=max(0.05, min(1.0, texture_keep)),
            feather_radius=max(0, feather_radius)
        )
        current_rgb = smoothed_rgb

    # --- Action 3: lighten skin on the smoothed result ---
    if include_lighten and mask_np is not None:
        lightened_rgb, light_alpha = calculate_tone_lift(
            img_rgb=current_rgb,
            skin_mask=mask_np,
            strength=max(0.0, min(1.0, strength)),
            base_tone_lab=parsed_lab,
            feather_radius=max(0, feather_radius)
        )
        la_f = (light_alpha.astype(np.float32) / 255.0)[:, :, None]
        current_rgb = np.clip(
            lightened_rgb.astype(np.float32) * la_f + current_rgb.astype(np.float32) * (1.0 - la_f),
            0, 255
        ).astype(np.uint8)

    out_buf = io.BytesIO()
    Image.fromarray(current_rgb).save(out_buf, format="JPEG", quality=88, optimize=True)
    total_time = time.time() - t_start
    logger.info(f"Preview rendered in {total_time:.3f}s ({w}x{h}, heal={include_heal}, lighten={include_lighten})")

    return Response(
        content=out_buf.getvalue(),
        media_type="image/jpeg",
        headers={
            "X-Process-Time": f"{total_time:.3f}",
            "X-Preview-Size": f"{w}x{h}",
            "X-Healed-Pixels": str(healed_pixels)
        }
    )


@app.post("/refine-point")
async def refine_point(
    image: UploadFile = File(..., description="Portrait image"),
    x: int = Form(..., description="X coordinate of click"),
    y: int = Form(..., description="Y coordinate of click"),
    action_type: str = Form("toggle", description="'toggle' | 'add_pimple' | 'sample_tone'"),
    blobs_json: str = Form("[]", description="JSON array of current blobs"),
    default_radius: int = Form(7, description="Default radius for newly added blemish")
):
    """
    Layer 4 Refinement:
    - action_type='toggle': toggles active state of nearest blob
    - action_type='add_pimple': adds a new pimple blob at (x, y)
    - action_type='sample_tone': samples RGB & LAB tone at (x, y)
    """
    try:
        existing_blobs = json.loads(blobs_json)
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request data: {e}")

    img_rgb = np.array(pil_image)
    h, w, _ = img_rgb.shape
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))

    sampled_tone = None
    toggled_blob = None
    new_blob = None

    if action_type == "sample_tone":
        # Sample average 5x5 patch around click
        y1, y2 = max(0, y - 2), min(h, y + 3)
        x1, x2 = max(0, x - 2), min(w, x + 3)
        patch_rgb = img_rgb[y1:y2, x1:x2]
        avg_rgb = np.mean(patch_rgb, axis=(0, 1)).astype(int).tolist()
        patch_lab = cv2.cvtColor(img_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2LAB)
        avg_lab = np.mean(patch_lab, axis=(0, 1)).astype(float).tolist()
        sampled_tone = {
            "rgb": avg_rgb,
            "lab": [round(v, 1) for v in avg_lab]
        }
        updated_blobs = existing_blobs

    elif action_type == "add_pimple":
        skin_mask = np.full((h, w), 255, dtype=np.uint8)
        updated_blobs, new_blob = add_blob_at_point(
            img_rgb=img_rgb,
            skin_mask=skin_mask,
            x=x,
            y=y,
            existing_blobs=existing_blobs,
            default_radius=default_radius
        )

    else:  # toggle
        updated_blobs, toggled_blob = toggle_blob_at_point(
            x=x,
            y=y,
            existing_blobs=existing_blobs
        )
        # If no blob was near enough to toggle, add a new one
        if toggled_blob is None:
            skin_mask = np.full((h, w), 255, dtype=np.uint8)
            updated_blobs, new_blob = add_blob_at_point(
                img_rgb=img_rgb,
                skin_mask=skin_mask,
                x=x,
                y=y,
                existing_blobs=existing_blobs,
                default_radius=default_radius
            )

    return JSONResponse({
        "status": "success",
        "action": action_type,
        "blobs": updated_blobs,
        "toggled_blob": toggled_blob,
        "new_blob": new_blob,
        "sampled_tone": sampled_tone
    })


@app.post("/refine-text")
async def refine_text(
    image: UploadFile = File(..., description="Portrait image"),
    prompt: str = Form(..., description="Natural language refinement instruction"),
    blobs_json: str = Form("[]", description="Current blobs JSON"),
    gemini_api_key: Optional[str] = Form(None, description="Gemini API Key")
):
    """
    Layer 4 Refinement via Natural Language text grounding.
    Sends image + prompt to Gemini Vision to locate bounding region and update blob list.
    """
    cfg = load_config()
    api_key = (gemini_api_key or "").strip() or cfg.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Gemini API Key is required for text refinement instructions.")

    try:
        existing_blobs = json.loads(blobs_json)
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request data: {e}")

    w, h = pil_image.size

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        img_buf = io.BytesIO()
        pil_image.save(img_buf, format="JPEG", quality=90)

        sys_prompt = f"""
You are an expert AI photo retouching assistant.
The user wants to refine blemish detection with this instruction: "{prompt}".

Locate the exact 2D bounding boxes [ymin, xmin, ymax, xmax] (normalized 0-1000) for the spots/blemishes/moles the user is referring to.
State whether the action is 'add' (include in removal) or 'remove' (ignore / exclude).

Output JSON schema:
{{
  "action": "add",
  "targets": [
    {{"box_2d": [ymin, xmin, ymax, xmax], "label": "blemish"}}
  ]
}}
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=img_buf.getvalue(), mime_type="image/jpeg"),
                sys_prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )

        res_data = json.loads(response.text)
        action_intent = res_data.get("action", "add")
        targets = res_data.get("targets", [])

        updated_blobs = list(existing_blobs)
        blob_id = max([b["id"] for b in existing_blobs], default=0) + 1

        for target in targets:
            box = target.get("box_2d") or target.get("box")
            if not box or len(box) != 4:
                continue
            ymin, xmin, ymax, xmax = box
            y1 = int((ymin / 1000.0) * h)
            x1 = int((xmin / 1000.0) * w)
            y2 = int((ymax / 1000.0) * h)
            x2 = int((xmax / 1000.0) * w)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            r = max(6, int(math.hypot(x2 - x1, y2 - y1) / 2.0) + 2)

            if action_intent == "remove":
                for b in updated_blobs:
                    bcx, bcy = b["centroid"]
                    if math.hypot(cx - bcx, cy - bcy) < (r + b.get("radius", 6)):
                        b["active"] = False
            else:
                updated_blobs.append({
                    "id": blob_id,
                    "bbox": [x1, y1, x2, y2],
                    "centroid": [round(cx, 1), round(cy, 1)],
                    "radius": r,
                    "area": int(math.pi * r * r),
                    "confidence": 0.95,
                    "active": True,
                    "label": "text_prompt",
                    "source": "text_grounding"
                })
                blob_id += 1

        return JSONResponse({
            "status": "success",
            "prompt": prompt,
            "action": action_intent,
            "blobs": updated_blobs
        })

    except Exception as e:
        logger.error(f"Text refinement error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text instruction refinement error: {e}")


@app.post("/apply-heal")
async def apply_heal(
    image: UploadFile = File(..., description="Portrait image"),
    blobs_json: Optional[str] = Form(None, description="JSON list of blobs to heal"),
    mask: Optional[UploadFile] = File(None, description="Optional binary mask image upload"),
    heal_mode: str = Form("full_inpaint", description="'full_inpaint' | 'calm_redness' | 'flatten_bump'"),
    texture_blend: float = Form(0.25, description="Skin texture blend ratio (0.0 to 1.0)"),
    feather_radius: int = Form(3, description="Feather blur radius in pixels"),
    grain_intensity: float = Form(0.03, description="Micro-grain intensity")
):
    """
    Layer 5 Action 1: Remove/Hide Pimples via Context-Tiled Inpainting or Dermatological Calming.
    Accepts active blobs JSON or binary mask PNG.
    Generates transparent RGBA PNG patch ready for non-destructive placement on Photoshop layer.
    """
    global lama_model
    t_start = time.time()
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    img_rgb = np.array(pil_image)
    h, w, _ = img_rgb.shape

    # Construct mask from blobs or uploaded mask
    if blobs_json:
        try:
            blobs = json.loads(blobs_json)
            mask_np = blobs_to_mask(blobs, (h, w), soft_falloff=False)
        except Exception as e:
            logger.warning(f"Failed to parse blobs_json: {e}")
            mask_np = np.zeros((h, w), dtype=np.uint8)
    elif mask is not None:
        mask_bytes = await mask.read()
        mask_pil = Image.open(io.BytesIO(mask_bytes))
        if mask_pil.mode == "RGBA":
            mask_np = np.array(mask_pil.split()[3])
        else:
            mask_np = np.array(mask_pil.convert("L"))
        if mask_np.shape[:2] != (h, w):
            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_np = np.where(mask_np > 10, 255, 0).astype(np.uint8)
    else:
        raise HTTPException(status_code=400, detail="Either blobs_json or mask file is required.")

    healed_count = int(np.sum(mask_np > 0))
    if healed_count == 0:
        transparent_empty = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        out_buf = io.BytesIO()
        transparent_empty.save(out_buf, format="PNG")
        return Response(content=out_buf.getvalue(), media_type="image/png", headers={"X-Healed-Pixels": "0"})

    # 1. Pre-process / Neutralize erythema & dark craters
    clean_rgb = neutralize_erythema(img_rgb, mask_np)

    if heal_mode == "calm_redness":
        # Calming mode: neutralizes angry redness and inflammation to natural skin tone
        final_rgb = clean_rgb
    elif heal_mode == "flatten_bump":
        # Frequency separation bump flattening
        orig_f = img_rgb.astype(np.float32)
        blurred_low = cv2.GaussianBlur(orig_f, (15, 15), 0)
        high_freq = orig_f - blurred_low
        # Neutralize low-frequency color & flatten high-frequency 3D bump
        clean_f = clean_rgb.astype(np.float32)
        mask_f = (mask_np.astype(np.float32) / 255.0)[:, :, None]
        final_rgb = np.clip(clean_f + high_freq * (1.0 - mask_f * 0.7), 0, 255).astype(np.uint8)
    else:
        # Default Full AI Inpainting Mode
        if lama_model is None:
            raise HTTPException(status_code=503, detail="AI inpainting model is not loaded.")

        inpainted_np = inpaint_with_context_tiling(
            model=lama_model,
            img_rgb=clean_rgb,
            mask_gray=mask_np,
            max_tile_size=768,
            context_pad=80
        )

        # 3. Healthy pore texture synthesis & illumination gradient alignment
        final_rgb = blend_skin_texture(
            original_img=img_rgb,
            inpainted_img=inpainted_np,
            mask_gray=mask_np,
            texture_blend=max(0.0, min(1.0, texture_blend)),
            grain_intensity=max(0.0, min(0.2, grain_intensity))
        )

    # 4. Feather mask edges
    feathered_alpha = apply_feather(mask_np, feather_radius=max(0, feather_radius))

    # 5. Build transparent RGBA PNG
    r = Image.fromarray(final_rgb[:, :, 0])
    g = Image.fromarray(final_rgb[:, :, 1])
    b = Image.fromarray(final_rgb[:, :, 2])
    a = Image.fromarray(feathered_alpha)
    rgba_result = Image.merge("RGBA", (r, g, b, a))

    out_buf = io.BytesIO()
    rgba_result.save(out_buf, format="PNG", optimize=True)
    total_time = time.time() - t_start

    logger.info(f"Apply-heal completed in {total_time:.3f}s (mode={heal_mode})")
    return Response(
        content=out_buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Process-Time": f"{total_time:.3f}",
            "X-Healed-Pixels": str(healed_count)
        }
    )


@app.post("/apply-smooth")
async def apply_smooth(
    image: UploadFile = File(..., description="Portrait image"),
    skin_mask: Optional[UploadFile] = File(None, description="Optional skin mask PNG"),
    strength: float = Form(0.45, description="Smoothing strength 0-1"),
    texture_keep: float = Form(0.4, description="Pore/texture retention 0-1"),
    feather_radius: int = Form(4, description="Edge feather radius")
):
    """
    Layer 5 Action: Smooth Skin.
    Frequency-separation smoothing + full-face redness evening within skin_mask.
    Returns a transparent RGBA patch for non-destructive placement in Photoshop.
    """
    t_start = time.time()
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    img_rgb = np.array(pil_image)
    h, w, _ = img_rgb.shape

    if skin_mask is not None:
        mask_bytes = await skin_mask.read()
        mask_pil = Image.open(io.BytesIO(mask_bytes))
        mask_np = np.array(mask_pil.split()[3]) if mask_pil.mode == "RGBA" else np.array(mask_pil.convert("L"))
        if mask_np.shape[:2] != (h, w):
            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        mask_np, _ = segment_face_skin(img_rgb, feather_radius=feather_radius)

    rgba_patch = create_smooth_rgba_patch(
        img_rgb=img_rgb,
        skin_mask=mask_np,
        strength=max(0.0, min(1.0, strength)),
        texture_keep=max(0.05, min(1.0, texture_keep)),
        feather_radius=max(0, feather_radius)
    )

    out_buf = io.BytesIO()
    rgba_patch.save(out_buf, format="PNG", optimize=True)
    total_time = time.time() - t_start
    logger.info(f"Apply-smooth completed in {total_time:.3f}s (strength={strength}, keep={texture_keep})")
    return Response(
        content=out_buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Process-Time": f"{total_time:.3f}",
            "X-Strength": str(strength)
        }
    )


@app.post("/apply-lighten")
async def apply_lighten(
    image: UploadFile = File(..., description="Portrait image"),
    skin_mask: Optional[UploadFile] = File(None, description="Optional skin mask PNG"),
    strength: float = Form(0.35, description="Lightening strength (0.0 to 1.0)"),
    base_tone_lab: Optional[str] = Form(None, description="Optional JSON [L, a, b] baseline"),
    feather_radius: int = Form(4, description="Edge feather radius")
):
    """
    Layer 5 Action 2: Lighten Skin.
    Applies tone-relative brightening boost to facial skin within skin_mask.
    Generates transparent RGBA PNG patch ready for non-destructive placement on Photoshop layer.
    """
    t_start = time.time()
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    img_rgb = np.array(pil_image)
    h, w, _ = img_rgb.shape

    parsed_lab = None
    if base_tone_lab:
        try:
            parsed_lab = json.loads(base_tone_lab)
        except Exception:
            pass

    if skin_mask is not None:
        mask_bytes = await skin_mask.read()
        mask_pil = Image.open(io.BytesIO(mask_bytes))
        if mask_pil.mode == "RGBA":
            mask_np = np.array(mask_pil.split()[3])
        else:
            mask_np = np.array(mask_pil.convert("L"))
        if mask_np.shape[:2] != (h, w):
            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        mask_np, _ = segment_face_skin(img_rgb, feather_radius=feather_radius)

    # Generate transparent lightened skin patch
    rgba_patch = create_lightened_rgba_patch(
        img_rgb=img_rgb,
        skin_mask=mask_np,
        strength=max(0.0, min(1.0, strength)),
        base_tone_lab=parsed_lab,
        feather_radius=feather_radius
    )

    out_buf = io.BytesIO()
    rgba_patch.save(out_buf, format="PNG", optimize=True)
    total_time = time.time() - t_start

    logger.info(f"Apply-lighten completed in {total_time:.3f}s with strength {strength}")
    return Response(
        content=out_buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Process-Time": f"{total_time:.3f}",
            "X-Strength": str(strength)
        }
    )


# =====================================================================
# LEGACY COMPATIBILITY ENDPOINTS (v1 Auto-Heal & Heal-Blemish)
# =====================================================================

@app.post("/auto-heal")
async def auto_heal(
    image: UploadFile = File(..., description="Portrait image for automated blemish removal"),
    sensitivity: float = Form(0.5, description="Detection sensitivity"),
    heal_mode: str = Form("full_inpaint", description="'full_inpaint' | 'calm_redness' | 'flatten_bump'"),
    texture_blend: float = Form(0.25, description="Skin texture blend ratio"),
    feather_radius: int = Form(3, description="Feather blur radius"),
    grain_intensity: float = Form(0.03, description="Micro-grain intensity"),
    dilate_radius: int = Form(3, description="Dilation radius"),
    gemini_api_key: Optional[str] = Form(None, description="Optional Gemini API key")
):
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    img_rgb = np.array(pil_image)
    skin_mask, _ = segment_face_skin(img_rgb)
    blobs, _ = detect_pimple_candidates(img_rgb, skin_mask, sensitivity=sensitivity)
    blobs_json = json.dumps(blobs)

    image.file.seek(0)

    return await apply_heal(
        image=image,
        blobs_json=blobs_json,
        mask=None,
        heal_mode=heal_mode,
        texture_blend=texture_blend,
        feather_radius=feather_radius,
        grain_intensity=grain_intensity
    )


@app.post("/detect-mask")
async def detect_mask_endpoint(
    image: UploadFile = File(..., description="Portrait image to extract blemish mask"),
    sensitivity: float = Form(0.5, description="Detection sensitivity"),
    dilate_radius: int = Form(3, description="Dilation radius"),
    gemini_api_key: Optional[str] = Form(None, description="Optional Gemini API key")
):
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    img_rgb = np.array(pil_image)
    skin_mask, _ = segment_face_skin(img_rgb)
    blobs, pimple_mask = detect_pimple_candidates(img_rgb, skin_mask, sensitivity=sensitivity)

    r = Image.new("L", pil_image.size, 255)
    g = Image.new("L", pil_image.size, 40)
    b = Image.new("L", pil_image.size, 40)
    rgba_mask = Image.merge("RGBA", (r, g, b, Image.fromarray(pimple_mask)))

    out_buf = io.BytesIO()
    rgba_mask.save(out_buf, format="PNG")
    return Response(content=out_buf.getvalue(), media_type="image/png")


@app.post("/heal-blemish")
async def heal_blemish(
    image: UploadFile = File(..., description="Cropped bounding box of the base portrait"),
    mask: UploadFile = File(..., description="Cropped bounding box of the painted blemish mask"),
    texture_blend: float = Form(0.25, description="High-frequency skin texture retention"),
    feather_radius: int = Form(3, description="Feather blur radius"),
    grain_intensity: float = Form(0.03, description="Micro-grain intensity")
):
    return await apply_heal(
        image=image,
        mask=mask,
        texture_blend=texture_blend,
        feather_radius=feather_radius,
        grain_intensity=grain_intensity
    )


if __name__ == "__main__":
    import argparse
    import socket
    import uvicorn

    parser = argparse.ArgumentParser(description="AI Retouching Backend Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host IP")
    parser.add_argument("--port", type=int, default=8765, help="Port number")
    args = parser.parse_args()

    def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) == 0

    selected_port = args.port
    if is_port_in_use(selected_port, args.host):
        for alt_port in [8766, 8001, 9001, 5005]:
            if not is_port_in_use(alt_port, args.host):
                selected_port = alt_port
                break

    logger.info(f"Starting server on http://{args.host}:{selected_port}")
    uvicorn.run("server:app", host=args.host, port=selected_port, reload=False)
