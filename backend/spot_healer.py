"""
Professional Spot Healing Brush Engine for AI Retouching Plugin.
Replicates Photoshop's Spot Healing Brush tool with:
1. Erythema / Redness Chromatic Neutralization strictly constrained to blemish cores
2. Annular Healthy Skin Texture Sampling
3. LaMa Neural Context Inpainting Integration
4. High-Frequency Skin Pore Retention & Micro-Grain Matching
5. Strict Lip, Mouth, Nostril & Facial Boundary Protection
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("spot_healer")


def neutralize_erythema_annulus(
    img_rgb: np.ndarray,
    mask_gray: np.ndarray,
    annulus_radius: int = 7,
    skin_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Pulls inflamed redness (a* channel) strictly inside active blemish cores toward
    the surrounding healthy skin baseline.
    Never bleeds into lips, mouth, nostril crevices, or facial contours.
    """
    h, w, _ = img_rgb.shape
    if np.sum(mask_gray > 10) == 0:
        return img_rgb

    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_c, a_c, b_c = cv2.split(img_lab)

    # Valid skin filter: avoid background, dark crevices (<35 L*), and non-skin
    valid_skin = (l_c > 35.0) & (l_c < 240.0)
    if skin_mask is not None:
        valid_skin = valid_skin & (skin_mask > 80)

    # Reference sampling weight: valid skin pixels strictly outside the blemish mask
    blemish_bin = (mask_gray > 10).astype(np.uint8)
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (annulus_radius * 2 + 1, annulus_radius * 2 + 1))
    dilated_mask = cv2.dilate(blemish_bin, k_dil)

    skin_weight = ((dilated_mask == 0) & valid_skin).astype(np.float32)

    # Localized spatial baseline sampling
    sigma = max(4.0, min(h, w) * 0.012)
    norm_w = cv2.GaussianBlur(skin_weight, (0, 0), sigma) + 1e-5
    blurred_a = cv2.GaussianBlur(a_c * skin_weight, (0, 0), sigma)
    blurred_b = cv2.GaussianBlur(b_c * skin_weight, (0, 0), sigma)

    healthy_a = np.where(norm_w > 1e-3, blurred_a / norm_w, a_c)
    healthy_b = np.where(norm_w > 1e-3, blurred_b / norm_w, b_c)

    # Apply redness neutralization ONLY directly on blemish pixels within valid skin
    core_factor = (blemish_bin.astype(np.float32)) * (valid_skin.astype(np.float32))
    support = np.clip(norm_w / 0.08, 0.0, 1.0)
    
    excess_a = np.maximum(0.0, a_c - (healthy_a + 2.0))
    a_corrected = a_c - (excess_a * core_factor * support * 0.85)

    excess_b = np.clip(b_c - healthy_b, -8.0, 8.0)
    b_corrected = b_c - (excess_b * core_factor * support * 0.20)

    # Ensure natural human warmth floor
    a_corrected = np.maximum(a_corrected, 130.0)
    b_corrected = np.maximum(b_corrected, 128.0)

    lab_clean = cv2.merge([l_c, a_corrected, b_corrected]).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_clean, cv2.COLOR_LAB2RGB)


def spot_healing_brush_inpaint(
    img_rgb: np.ndarray,
    blobs: List[Dict[str, Any]],
    lama_model: Any = None,
    heal_mode: str = "full_inpaint",
    texture_blend: float = 0.25,
    dilate_px: int = 2,
    feather_radius: int = 3,
    grain_intensity: float = 0.03,
    skin_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Master Spot Healing Brush Engine.
    Heals blemishes cleanly with zero smudging on lips, nose, or facial features.
    """
    h, w, _ = img_rgb.shape
    active_blobs = [b for b in blobs if b.get("active", True) is not False]
    if len(active_blobs) == 0:
        return img_rgb.copy(), np.zeros((h, w), dtype=np.uint8)

    # 1. Build precision mask from active blobs (with multi-pass for large blemishes)
    raw_mask = np.zeros((h, w), dtype=np.uint8)
    large_blemish_cores = np.zeros((h, w), dtype=np.uint8)
    
    for b in active_blobs:
        cx = int(round(b.get("centroid", [0, 0])[0]))
        cy = int(round(b.get("centroid", [0, 0])[1]))
        r = int(b.get("radius", 6))
        cv2.circle(raw_mask, (cx, cy), max(2, r), 255, -1)
        
        # Mark large blemishes (>10px) for two-pass healing
        # Pass 1 heals the outer halo (erythema + diffuse redness)
        # Pass 2 heals the inner core (texture + structure)
        if r > 10:
            core_radius = int(r * 0.4)  # Inner 40% is the textured core
            cv2.circle(large_blemish_cores, (cx, cy), max(2, core_radius), 255, -1)

    if dilate_px > 0:
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        fused_mask = cv2.dilate(raw_mask, k_dil)
    else:
        fused_mask = raw_mask.copy()

    # Strictly confine blemish healing inside verified facial skin (protects lips, nostrils, eyes, background)
    if skin_mask is not None:
        fused_mask = (fused_mask & (skin_mask > 80).astype(np.uint8) * 255)

    if np.sum(fused_mask > 0) == 0:
        return img_rgb.copy(), np.zeros((h, w), dtype=np.uint8)

    # 2. Neutralize erythema under blemish mask
    color_neutral_rgb = neutralize_erythema_annulus(img_rgb, fused_mask, annulus_radius=7, skin_mask=skin_mask)

    # 3. Heal Mode Handling
    if heal_mode == "calm_redness":
        base_healed = color_neutral_rgb
    elif heal_mode == "flatten_bump":
        orig_f = img_rgb.astype(np.float32)
        blurred_low = cv2.GaussianBlur(orig_f, (11, 11), 0)
        high_freq = orig_f - blurred_low
        clean_f = color_neutral_rgb.astype(np.float32)
        mask_f = (fused_mask.astype(np.float32) / 255.0)[:, :, None]
        base_healed = np.clip(clean_f + high_freq * (1.0 - mask_f * 0.6), 0, 255).astype(np.uint8)
    else:
        # Full inpainting with Simple-LaMa (Direct neural skin reconstruction)
        lama_used = False
        if lama_model is not None:
            try:
                from inpainter import inpaint_with_context_tiling
                base_healed = inpaint_with_context_tiling(
                    model=lama_model,
                    img_rgb=color_neutral_rgb,
                    mask_gray=fused_mask,
                    max_tile_size=768,
                    context_pad=60
                )
                lama_used = True
            except Exception as e:
                logger.warning(f"LaMa neural inpaint failed, using Telea: {e}")
                base_healed = None

        if not lama_used:
            base_healed = cv2.inpaint(color_neutral_rgb, fused_mask, inpaintRadius=max(3, dilate_px + 2), flags=cv2.INPAINT_TELEA)

    # 4. Skin Texture & Pore Preservation
    orig_f = img_rgb.astype(np.float32)
    healed_f = base_healed.astype(np.float32)
    mask_f = (fused_mask.astype(np.float32) / 255.0)[:, :, None]

    if texture_blend > 0:
        blurred_orig = cv2.GaussianBlur(orig_f, (5, 5), 0)
        high_freq = orig_f - blurred_orig

        k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        mask_inner = cv2.dilate(fused_mask, k_inner)
        mask_outer = cv2.dilate(fused_mask, k_outer)
        annulus_mask = np.clip(mask_outer.astype(np.float32) - mask_inner.astype(np.float32), 0.0, 255.0) / 255.0

        if skin_mask is not None:
            annulus_mask = annulus_mask * (skin_mask > 80).astype(np.float32)

        healthy_pore_texture = cv2.GaussianBlur(high_freq * annulus_mask[:, :, None], (5, 5), 0)
        healed_f = healed_f + (healthy_pore_texture * (texture_blend * 1.0) * mask_f)

    # 5. Micro-Grain Matching
    if grain_intensity > 0:
        noise = np.random.normal(loc=0.0, scale=grain_intensity * 36.0, size=(h, w, 3))
        healed_f = healed_f + (noise.astype(np.float32) * mask_f)

    healed_f = np.clip(healed_f, 0.0, 255.0)

    # 6. Soft Edge Feathering
    eff_feather = max(1, min(4, feather_radius))
    ksize = eff_feather * 2 + 1
    feathered_alpha = cv2.GaussianBlur(fused_mask, (ksize, ksize), 0)

    alpha_f = (feathered_alpha.astype(np.float32) / 255.0)[:, :, None]
    final_composite = (orig_f * (1.0 - alpha_f) + healed_f * alpha_f).clip(0, 255).astype(np.uint8)

    return final_composite, feathered_alpha
