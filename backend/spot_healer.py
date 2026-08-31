"""
Professional Spot Healing Brush Engine for AI Retouching Plugin.
Replicates and enhances Photoshop's Spot Healing Brush tool with:
1. Erythema / Redness Chromatic Neutralization
2. Annular Healthy Skin Texture Sampling
3. Multi-Scale Navier-Stokes / Telea Inpainting + Poisson Texture Transfer
4. LaMa Neural Context Inpainting Integration
5. Seamless 3D Bump Flattening & Illumination Gradient Matching
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
    annulus_radius: int = 12,
    skin_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Pulls inflamed redness (a* channel) inside the blemish mask toward the healthy
    surrounding skin baseline sampled from the local skin annulus.

    Strictly validates skin pixels (excluding background whites/blacks, nostrils, and hair)
    so edge features like the nose tip and chin never sample background gray values.
    Only neutralizes EXCESS redness (a* > healthy baseline); never desaturates normal skin.
    """
    h, w, _ = img_rgb.shape
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_c, a_c, b_c = cv2.split(img_lab)

    # Valid skin filter: avoid pure white background (>250), dark crevices (<20), and non-skin
    valid_skin = (l_c > 25.0) & (l_c < 248.0)
    if skin_mask is not None:
        valid_skin = valid_skin & (skin_mask > 25)

    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (annulus_radius * 2 + 1, annulus_radius * 2 + 1))
    dilated_mask = cv2.dilate(mask_gray, k_dil)

    # Reference sampling weight: valid skin pixels strictly outside the blemish mask
    skin_weight = ((dilated_mask == 0) & valid_skin).astype(np.float32)

    # Localized spatial sigma to prevent distant background bleeding
    sigma = max(6.0, min(h, w) * 0.025)
    norm_w = cv2.GaussianBlur(skin_weight, (0, 0), sigma)
    blurred_a = cv2.GaussianBlur(a_c * skin_weight, (0, 0), sigma)
    blurred_b = cv2.GaussianBlur(b_c * skin_weight, (0, 0), sigma)

    # Safe fallback to current values if no local skin reference exists
    healthy_a = np.where(norm_w > 1e-3, blurred_a / (norm_w + 1e-6), a_c)
    healthy_b = np.where(norm_w > 1e-3, blurred_b / (norm_w + 1e-6), b_c)

    # Natural skin chroma floor: OpenCV LAB neutral is 128. Real human skin has a* >= 134, b* >= 132.
    healthy_a = np.clip(healthy_a, 134.0, 175.0)
    healthy_b = np.clip(healthy_b, 132.0, 180.0)

    # Only pull down EXCESS redness where current a* is significantly higher than healthy baseline
    mask_factor = (dilated_mask.astype(np.float32) / 255.0)
    support = np.clip(norm_w / 0.08, 0.0, 1.0)
    
    excess_a = np.maximum(0.0, a_c - healthy_a)
    a_corrected = a_c - (excess_a * mask_factor * support * 0.85)

    # Keep b* warm (never pull toward cold/blue)
    excess_b_dev = (b_c - healthy_b) * mask_factor * support * 0.35
    b_corrected = b_c - excess_b_dev

    # Safety clamp: never allow a* or b* to fall into gray/green/blue territory on skin
    a_corrected = np.maximum(a_corrected, 132.0)
    b_corrected = np.maximum(b_corrected, 130.0)

    lab_clean = cv2.merge([l_c, a_corrected, b_corrected]).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_clean, cv2.COLOR_LAB2RGB)


def match_tone_and_texture(
    original_img: np.ndarray,
    healed_img: np.ndarray,
    mask_gray: np.ndarray,
    blend: float = 0.5,
    skin_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Component-Localized Tone & Texture Normalizer.

    Adjusts each connected healed patch locally against its immediate surrounding
    skin annulus (10-30px) rather than a whole-image global average, preserving the
    distinct lighting and 3D volume of the nose, chin, and cheekbones.
    """
    mask_bin = (mask_gray > 10).astype(np.uint8)
    if np.sum(mask_bin) == 0:
        return healed_img

    h, w = mask_gray.shape[:2]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin)
    if num_labels <= 1:
        return healed_img

    orig_lab = cv2.cvtColor(original_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    healed_lab = cv2.cvtColor(healed_img, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Valid skin mask
    valid_skin = (orig_lab[:, :, 0] > 25.0) & (orig_lab[:, :, 0] < 248.0)
    if skin_mask is not None:
        valid_skin = valid_skin & (skin_mask > 25)

    # Iterate over individual connected components for localized matching
    for i in range(1, num_labels):
        comp_mask = (labels == i).astype(np.uint8)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 4:
            continue

        # Local ring around this specific spot
        k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        local_ring = ((cv2.dilate(comp_mask, k_out) - cv2.dilate(comp_mask, k_in)) > 0) & valid_skin & (mask_bin == 0)

        if np.sum(local_ring) < 30:
            continue

        # Local luminance & chroma stats
        ring_l = orig_lab[local_ring, 0]
        core_l = healed_lab[comp_mask > 0, 0]
        ring_l_mean = float(np.median(ring_l))
        core_l_mean = float(np.median(core_l))

        # Gentle localized luminance offset (clamped to max +/- 4.0 LAB units to prevent smudging)
        l_offset = np.clip((ring_l_mean - core_l_mean) * blend, -4.0, 4.0)

        # Soft blend mask
        soft = cv2.GaussianBlur(comp_mask.astype(np.float32), (7, 7), 0)
        healed_lab[:, :, 0] = healed_lab[:, :, 0] + (l_offset * soft)

        # Chroma matching (light touch)
        for ch in (1, 2):
            ring_ch = float(np.median(orig_lab[local_ring, ch]))
            core_ch = float(np.median(healed_lab[comp_mask > 0, ch]))
            ch_offset = np.clip((ring_ch - core_ch) * 0.4 * blend, -3.0, 3.0)
            healed_lab[:, :, ch] = healed_lab[:, :, ch] + (ch_offset * soft)

    out = cv2.cvtColor(healed_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out


def spot_healing_brush_inpaint(
    img_rgb: np.ndarray,
    blobs: List[Dict[str, Any]],
    lama_model: Any = None,
    heal_mode: str = "pro_spot_healer",
    texture_blend: float = 0.30,
    dilate_px: int = 4,
    feather_radius: int = 3,
    grain_intensity: float = 0.03,
    skin_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Master Spot Healing Brush Engine:
    Completely eliminates pimples, red papules, whiteheads, blackheads, and dark craters.

    Args:
        skin_mask: optional facial-skin segmentation (uint8 HxW). Restricts the
            erythema baseline and tone-matching ring to real skin pixels so
            heals beside nostrils / nose folds don't sample dark crevice
            pixels (the cause of grey patches and reddish residue on the nose).

    Returns:
        healed_rgb: uint8 HxWx3 fully composited image
        transparent_alpha: uint8 HxW feathered mask for non-destructive Photoshop placement
    """
    h, w, _ = img_rgb.shape
    active_blobs = [b for b in blobs if b.get("active", True) is not False]
    if len(active_blobs) == 0:
        return img_rgb.copy(), np.zeros((h, w), dtype=np.uint8)

    # 1. Build Generous Spot Healing Mask (1.4x radius expansion to encompass redness halo)
    raw_mask = np.zeros((h, w), dtype=np.uint8)
    for b in active_blobs:
        cx = int(round(b.get("centroid", [0, 0])[0]))
        cy = int(round(b.get("centroid", [0, 0])[1]))
        r = int(b.get("radius", 6))
        expanded_r = max(5, int(r * 1.35) + 2)
        cv2.circle(raw_mask, (cx, cy), expanded_r, 255, -1)

    # 2. Morphological Cluster Fusion: bridges clustered acne into smooth contiguous contours
    k_fuse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fused_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, k_fuse)
    if dilate_px > 0:
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        fused_mask = cv2.dilate(fused_mask, k_dil)

    # 3. Step 1: Neutralize Erythema & Deep Redness under mask with strict skin safety
    color_neutral_rgb = neutralize_erythema_annulus(img_rgb, fused_mask, annulus_radius=12, skin_mask=skin_mask)

    # 4. Step 2: Neural Inpainting (LaMa) with Telea as fallback only.
    lama_used = False
    if lama_model is not None:
        try:
            from inpainter import inpaint_with_context_tiling
            base_healed = inpaint_with_context_tiling(
                model=lama_model,
                img_rgb=color_neutral_rgb,
                mask_gray=fused_mask,
                max_tile_size=768,
                context_pad=80
            )
            lama_used = True
        except Exception as e:
            logger.warning(f"LaMa neural inpaint failed, falling back to Telea: {e}")
            base_healed = None
    else:
        base_healed = None

    if not lama_used:
        base_healed = cv2.inpaint(color_neutral_rgb, fused_mask, inpaintRadius=max(5, dilate_px + 4), flags=cv2.INPAINT_TELEA)

    # 5. Step 3: High-Frequency Annular Skin Pore Synthesis (Photoshop Texture Matching)
    orig_f = img_rgb.astype(np.float32)
    healed_f = base_healed.astype(np.float32)
    mask_f = (fused_mask.astype(np.float32) / 255.0)[:, :, None]

    if texture_blend > 0:
        # Extract genuine high-frequency micro-pores from clean surrounding skin
        blurred_orig = cv2.GaussianBlur(orig_f, (5, 5), 0)
        high_freq = orig_f - blurred_orig

        k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        mask_inner = cv2.dilate(fused_mask, k_inner)
        mask_outer = cv2.dilate(fused_mask, k_outer)
        annulus_mask = np.clip(mask_outer.astype(np.float32) - mask_inner.astype(np.float32), 0.0, 255.0) / 255.0

        if skin_mask is not None:
            annulus_mask = annulus_mask * (skin_mask > 25).astype(np.float32)

        healthy_pore_texture = cv2.GaussianBlur(high_freq * annulus_mask[:, :, None], (5, 5), 0)
        healed_f = healed_f + (healthy_pore_texture * (texture_blend * 1.1) * mask_f)

    # 6. Step 4: Micro-Grain Matching (subtle: 2-3% -> sigma ~1.5 levels)
    if grain_intensity > 0:
        noise = np.random.normal(loc=0.0, scale=grain_intensity * 48.0, size=(h, w, 3))
        healed_f = healed_f + (noise.astype(np.float32) * mask_f)

    healed_f = np.clip(healed_f, 0.0, 255.0).astype(np.uint8)

    # 6.5 Step 4.5: Localized Tone Normalization
    try:
        healed_f = match_tone_and_texture(img_rgb, healed_f, fused_mask, blend=0.4, skin_mask=skin_mask)
    except Exception as e:
        logger.warning(f"Tone matching skipped: {e}")

    healed_f = healed_f.astype(np.float32)

    # 7. Step 5: Soft Edge Feathering (resolution-adaptive)
    eff_feather = max(feather_radius, int(min(h, w) * 0.003))
    if eff_feather > 0:
        ksize = eff_feather * 2 + 1
        feathered_alpha = cv2.GaussianBlur(fused_mask, (ksize, ksize), 0)
    else:
        feathered_alpha = fused_mask.copy()

    alpha_f = (feathered_alpha.astype(np.float32) / 255.0)[:, :, None]
    final_composite = (orig_f * (1.0 - alpha_f) + healed_f * alpha_f).clip(0, 255).astype(np.uint8)

    return final_composite, feathered_alpha
