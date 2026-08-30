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
    annulus_radius: int = 15,
    skin_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Pulls inflamed redness (a* channel) inside the blemish mask toward the healthy
    surrounding skin baseline sampled from the outer annulus.

    When skin_mask is provided, only pixels classified as facial skin contribute
    to the healthy baseline. Heals beside the nostrils / nose folds otherwise
    sample dark crevice pixels, skewing the baseline and leaving grey
    (over-desaturated) or reddish (mis-targeted) patches behind.
    """
    h, w, _ = img_rgb.shape
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_c, a_c, b_c = cv2.split(img_lab)

    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (annulus_radius, annulus_radius))
    dilated_mask = cv2.dilate(mask_gray, k_dil)
    skin_weight = (dilated_mask == 0).astype(np.float32)
    if skin_mask is not None:
        skin_weight = skin_weight * (skin_mask > 30).astype(np.float32)

    sigma = max(12.0, min(h, w) * 0.08)
    norm_w = cv2.GaussianBlur(skin_weight, (0, 0), sigma)
    blurred_a = cv2.GaussianBlur(a_c * skin_weight, (0, 0), sigma)
    blurred_b = cv2.GaussianBlur(b_c * skin_weight, (0, 0), sigma)

    # Reliable-reference guard: where too few healthy reference pixels surround
    # the blemish (nostril borders, oversized masks, image edges) the weighted
    # baseline degenerates. Fade the chroma correction out there instead of
    # pulling a* toward a meaningless near-zero average (green/grey cast).
    support = np.clip(norm_w / 0.10, 0.0, 1.0)
    healthy_a = np.where(norm_w > 1e-3, blurred_a / (norm_w + 1e-6), a_c)
    healthy_b = np.where(norm_w > 1e-3, blurred_b / (norm_w + 1e-6), b_c)

    # 0.92 pull on a*: inflamed red must be almost fully replaced by the
    # healthy baseline or a faint pink ghost survives under the heal.
    mask_factor = (dilated_mask.astype(np.float32) / 255.0) * support
    a_pull = mask_factor * 0.92
    b_pull = mask_factor * 0.70
    a_corrected = a_c * (1.0 - a_pull) + healthy_a * a_pull
    b_corrected = b_c * (1.0 - b_pull) + healthy_b * b_pull

    lab_clean = cv2.merge([l_c, a_corrected, b_corrected]).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_clean, cv2.COLOR_LAB2RGB)


def match_tone_and_texture(
    original_img: np.ndarray,
    healed_img: np.ndarray,
    mask_gray: np.ndarray,
    blend: float = 0.9,
    skin_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Healed-Patch Tone & Texture Normalizer (the 'invisible patch' pass).

    LaMa reconstructs plausible skin, but its luminance mean, contrast, and pore
    amplitude can deviate slightly from the surrounding skin, leaving a faintly
    visible patch. This pass statistically matches the healed core to the healthy
    ring around it — the exact adjustment a professional retoucher makes manually:

    1. L* mean shift   -> kills darker/lighter patches
    2. L* contrast gain -> matches tonal energy of surrounding skin
    3. a*/b* chroma shift -> matches skin hue (light touch; erythema handled earlier)
    4. High-frequency amplitude equalization -> pore density matches neighbors

    When skin_mask is provided, only pixels classified as facial skin form the
    healthy ring. Nose-adjacent heals otherwise match against nostril shadows
    and nose-fold pixels — the source of grey patches on the nose. All tone
    corrections are additionally localized to the healed core so surrounding
    features (nostrils, lips, hair) are never remapped at the feather band.
    """
    mask_bin = mask_gray > 10
    if np.sum(mask_bin) == 0:
        return healed_img

    # Guard: if the mask covers most of the frame there is no representative
    # healthy ring left to match against — matching would corrupt global tone.
    h, w = mask_gray.shape[:2]
    if np.sum(mask_bin) > 0.35 * h * w:
        return healed_img

    # Healthy ring: band of real skin 10-60px outside the mask
    mask_u8 = mask_bin.astype(np.uint8)
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121))
    ring = ((cv2.dilate(mask_u8, k_out) - cv2.dilate(mask_u8, k_in)) > 0) & ~mask_bin
    # Skin-only ring sampling: exclude nostril shadows, folds, hair and
    # background pixels from the reference statistics.
    if skin_mask is not None:
        ring = ring & (skin_mask > 30)
    if np.sum(ring) < 500:
        # Not enough classified healthy skin around the patch to match against
        # (e.g. spot at the nostril border) — keep LaMa output untouched rather
        # than matching against non-skin statistics.
        return healed_img

    orig_lab = cv2.cvtColor(original_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    healed_lab = cv2.cvtColor(healed_img, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Soft application mask: corrections fade out within a few pixels of the
    # patch edge so the downstream feathered composite blends raw LaMa values
    # (which already match the context) at the boundary instead of remapped ones.
    soft = cv2.GaussianBlur(mask_bin.astype(np.float32), (11, 11), 0)

    # 1 + 2. Luminance mean & contrast matching
    ring_l = orig_lab[ring, 0]
    core_l = healed_lab[mask_bin, 0]
    ring_mean, ring_std = float(ring_l.mean()), float(ring_l.std() + 1e-5)
    core_mean, core_std = float(core_l.mean()), float(core_l.std() + 1e-5)

    gain = float(np.clip(ring_std / core_std, 0.75, 1.35))
    l_chan = healed_lab[:, :, 0]
    l_adjusted = np.clip(
        (l_chan - core_mean) * gain + core_mean + (ring_mean - core_mean) * blend,
        0, 255
    )
    healed_lab[:, :, 0] = l_chan * (1.0 - soft) + l_adjusted * soft

    # 3. Chroma shift toward ring tone (subtle). MEDIAN, not mean: inflamed
    # halo pixels around a fresh pimple would drag the mean toward red and
    # re-introduce the exact redness we just removed.
    for ch in (1, 2):
        ring_m = float(np.median(orig_lab[ring, ch]))
        core_m = float(np.median(healed_lab[mask_bin, ch]))
        ch_adjusted = np.clip(
            healed_lab[:, :, ch] + (ring_m - core_m) * 0.55 * blend, 0, 255
        )
        healed_lab[:, :, ch] = healed_lab[:, :, ch] * (1.0 - soft) + ch_adjusted * soft

    out = cv2.cvtColor(healed_lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    # 4. High-frequency pore amplitude equalization (per channel).
    # Replace (not add) the AC component so amplitude matches the ring.
    healed_f = healed_img.astype(np.float32)
    orig_f = original_img.astype(np.float32)
    hf_healed = healed_f - cv2.GaussianBlur(healed_f, (5, 5), 0)
    hf_orig = orig_f - cv2.GaussianBlur(orig_f, (5, 5), 0)

    for ch in range(3):
        amp_ring = float(hf_orig[ring, ch].std() + 1e-5)
        amp_core = float(hf_healed[mask_bin, ch].std() + 1e-5)
        ratio = float(np.clip(amp_ring / amp_core, 0.7, 1.4))
        if abs(ratio - 1.0) < 0.05:
            continue
        hf_ch = hf_healed[:, :, ch]
        # delta replaces the existing AC amplitude with the matched one
        hf_healed[:, :, ch] = hf_ch * ratio

    out = out + (hf_healed - (healed_f - cv2.GaussianBlur(healed_f, (5, 5), 0))) * mask_bin[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def spot_healing_brush_inpaint(
    img_rgb: np.ndarray,
    blobs: List[Dict[str, Any]],
    lama_model: Any = None,
    heal_mode: str = "pro_spot_healer",
    texture_blend: float = 0.35,
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
        expanded_r = max(5, int(r * 1.45) + 2)
        cv2.circle(raw_mask, (cx, cy), expanded_r, 255, -1)

    # 2. Morphological Cluster Fusion: bridges clustered acne into smooth contiguous contours
    k_fuse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fused_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, k_fuse)
    if dilate_px > 0:
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        fused_mask = cv2.dilate(fused_mask, k_dil)

    # 3. Step 1: Neutralize Erythema & Deep Redness under mask
    color_neutral_rgb = neutralize_erythema_annulus(img_rgb, fused_mask, annulus_radius=15, skin_mask=skin_mask)

    # 4. Step 2: Neural Inpainting (LaMa) with Telea as fallback only.
    # NOTE: never run Telea as a pre-pass when LaMa is available — diffusion
    # inpainting flattens skin into smudges. LaMa reconstructs real skin.
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

        k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        mask_inner = cv2.dilate(fused_mask, k_inner)
        mask_outer = cv2.dilate(fused_mask, k_outer)
        annulus_mask = np.clip(mask_outer.astype(np.float32) - mask_inner.astype(np.float32), 0.0, 255.0) / 255.0

        healthy_pore_texture = cv2.GaussianBlur(high_freq * annulus_mask[:, :, None], (5, 5), 0)
        healed_f = healed_f + (healthy_pore_texture * (texture_blend * 1.35) * mask_f)

    # 6. Step 4: Micro-Grain Matching (subtle: 3% -> sigma ~2 levels)
    if grain_intensity > 0:
        noise = np.random.normal(loc=0.0, scale=grain_intensity * 64.0, size=(h, w, 3))
        healed_f = healed_f + (noise.astype(np.float32) * mask_f)

    healed_f = np.clip(healed_f, 0.0, 255.0).astype(np.uint8)

    # 6.5 Step 4.5: Tone & Texture Normalization — statistically match each
    # healed region's luminance, contrast, chroma and pore amplitude to the
    # healthy skin ring around it so the patch becomes invisible.
    try:
        healed_f = match_tone_and_texture(img_rgb, healed_f, fused_mask, blend=0.9, skin_mask=skin_mask)
    except Exception as e:
        logger.warning(f"Tone matching skipped: {e}")

    healed_f = healed_f.astype(np.float32)

    # 7. Step 5: Soft Edge Feathering (resolution-adaptive so full-resolution
    # camera files get Photoshop-style soft transitions, not razor edges)
    eff_feather = max(feather_radius, int(min(h, w) * 0.004))
    if eff_feather > 0:
        ksize = eff_feather * 2 + 1
        feathered_alpha = cv2.GaussianBlur(fused_mask, (ksize, ksize), 0)
    else:
        feathered_alpha = fused_mask.copy()

    alpha_f = (feathered_alpha.astype(np.float32) / 255.0)[:, :, None]
    final_composite = (orig_f * (1.0 - alpha_f) + healed_f * alpha_f).clip(0, 255).astype(np.uint8)

    return final_composite, feathered_alpha
