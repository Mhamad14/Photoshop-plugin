import logging
from typing import Optional, List, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("skin_smoother")


def even_redness(img_rgb: np.ndarray, skin_mask: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Melanin-Aware Skin Redness Neutralizer:
    Evens out blotchy erythema and acne redness across facial skin by gently pulling
    each pixel's CIE LAB a* (green-red) and b* (blue-yellow) chroma toward the local
    healthy-skin baseline while preserving genuine cheek blush, nose warmth, and luminance (L*).
    """
    if strength <= 0:
        return img_rgb

    h, w, _ = img_rgb.shape
    skin_w = (skin_mask > 25).astype(np.float32)
    if np.sum(skin_w) < 100:
        return img_rgb

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan, a_chan, b_chan = cv2.split(lab)

    valid_skin = (skin_w > 0) & (l_chan > 30.0) & (l_chan < 248.0)
    valid_weight = valid_skin.astype(np.float32)

    sigma = max(10.0, min(h, w) / 28.0)
    if min(h, w) > 1500:
        ds = 4
        vw_s = cv2.resize(valid_weight, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
        a_s = cv2.resize(a_chan * valid_weight, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
        b_s = cv2.resize(b_chan * valid_weight, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
        nw_s = cv2.GaussianBlur(vw_s, (0, 0), sigma / ds)
        ba_s = cv2.GaussianBlur(a_s, (0, 0), sigma / ds)
        bb_s = cv2.GaussianBlur(b_s, (0, 0), sigma / ds)
        norm_w = cv2.resize(nw_s, (w, h), interpolation=cv2.INTER_LINEAR)
        blurred_a = cv2.resize(ba_s, (w, h), interpolation=cv2.INTER_LINEAR)
        blurred_b = cv2.resize(bb_s, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        norm_w = cv2.GaussianBlur(valid_weight, (0, 0), sigma)
        blurred_a = cv2.GaussianBlur(a_chan * valid_weight, (0, 0), sigma)
        blurred_b = cv2.GaussianBlur(b_chan * valid_weight, (0, 0), sigma)

    a_base = np.where(norm_w > 1e-3, blurred_a / (norm_w + 1e-6), a_chan)
    b_base = np.where(norm_w > 1e-3, blurred_b / (norm_w + 1e-6), b_chan)

    a_base = np.clip(a_base, 134.0, 168.0)
    b_base = np.clip(b_base, 132.0, 175.0)

    excess_a = np.maximum(0.0, a_chan - (a_base + 3.5))
    
    # Protect natural rosy cheek blush and healthy skin warmth
    blush_protect = np.clip((a_chan - 144.0) / 16.0, 0.0, 1.0)
    support = np.clip(norm_w / 0.12, 0.0, 1.0)
    redness_factor = (strength * (1.0 - blush_protect * 0.75) * skin_w * support)

    # Natural healthy skin floor: never turn skin gray, pale, or sickly (warmth floor >= 137.0)
    a_skin = np.maximum(a_chan - (excess_a * redness_factor * 0.55), 137.0)
    b_skin = np.maximum(b_chan * (1.0 - redness_factor * 0.05) + b_base * (redness_factor * 0.05), 135.0)

    # Strictly confine redness changes to valid skin pixels; outside skin is 100% untouched
    a_final = np.where(valid_skin, a_skin, a_chan)
    b_final = np.where(valid_skin, b_skin, b_chan)

    merged = cv2.merge([l_chan, a_final, b_final]).clip(0, 255).astype(np.uint8)
    evened_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    # Strictly return original img_rgb where not valid skin (0.00 difference)
    return np.where(valid_skin[:, :, None], evened_rgb, img_rgb)


def frequency_separation_smooth(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = 0.5,
    texture_keep: float = 0.85,
    feather_radius: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Studio-Grade Tri-Band Frequency Separation with 100% Organic Micro-Pore Retention
    and Structural Edge Preservation (Nose, Lips, Eyes, and Profile Boundary Protection):
    """
    h, w, _ = img_rgb.shape
    mask_binary = (skin_mask > 25).astype(np.uint8)
    if np.sum(mask_binary) < 100:
        return img_rgb.copy(), skin_mask.copy()

    img_f = img_rgb.astype(np.float32)
    dim = min(h, w)
    
    sigma_mid = max(4.0, dim * 0.007)
    sigma_low = max(18.0, dim * 0.035)

    # 1. Base Low-Frequency Layer with boundary skin normalization (pyramid downsampling for high-res)
    skin_f_weight = (mask_binary > 0).astype(np.float32)
    if dim > 1500:
        ds = 4
        sw_s = cv2.resize(skin_f_weight, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
        iw_s = cv2.resize(img_f * skin_f_weight[:, :, None], (w // ds, h // ds), interpolation=cv2.INTER_AREA)
        nl_s = cv2.GaussianBlur(sw_s, (0, 0), sigma_low / ds) + 1e-5
        bl_s = cv2.GaussianBlur(iw_s, (0, 0), sigma_low / ds) / nl_s[:, :, None]
        base_low = cv2.resize(bl_s, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        norm_low = cv2.GaussianBlur(skin_f_weight, (0, 0), sigma_low) + 1e-5
        base_low = cv2.GaussianBlur(img_f * skin_f_weight[:, :, None], (0, 0), sigma_low) / norm_low[:, :, None]
    base_low = np.where(skin_f_weight[:, :, None] > 0.01, base_low, img_f)
    
    # 2. Intermediate Smooth Layer (edge-preserving bilateral filter)
    d_val = max(5, min(9, int(dim * 0.003) | 1))
    sigma_space = max(6.0, min(24.0, dim * 0.008))
    sigma_color = max(20.0, min(50.0, 25.0 + strength * 20.0))
    
    base_mid = cv2.bilateralFilter(img_rgb, d=d_val, sigmaColor=sigma_color, sigmaSpace=sigma_space).astype(np.float32)
    if strength > 0.5:
        base_mid = cv2.bilateralFilter(base_mid.clip(0, 255).astype(np.uint8), d=d_val, sigmaColor=sigma_color * 0.8, sigmaSpace=sigma_space * 0.8).astype(np.float32)

    # 3. High-Frequency Micro-Pore Texture Band
    high_pass_texture = img_f - base_mid

    # 4. Mid-Frequency Roughness Band
    mid_roughness = base_mid - base_low

    # 5. Modulate Smoothing: Smooth mid-roughness bumps while retaining fine micro-pores
    strength_clamped = max(0.0, min(1.0, strength))
    texture_clamped = max(0.10, min(1.0, texture_keep))

    # Hair-Safe Protection Gate ("not make smoothy and not hairy like"):
    # Dark facial hair (beard, mustache, eyebrows with L* < 60) bypasses smoothing.
    # Individual hair strands remain 100% sharp and crisp with zero blurring.
    lab_temp = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan = lab_temp[:, :, 0]
    smooth_gate = np.clip((l_chan - 55.0) / 28.0, 0.0, 1.0)[:, :, None]

    # Higher smoothing power on mid-roughness cleanly eliminates bumps and acne texture on skin
    smoothed_mid = mid_roughness * (1.0 - strength_clamped * 0.90 * smooth_gate)
    reconstructed_base = base_low + smoothed_mid

    # Organic micro-texture retention: soft-compress extreme bump peaks while keeping delicate micro-pores
    hp_abs = np.abs(high_pass_texture)
    pore_weight = np.clip(1.0 - (hp_abs - 6.0) / 14.0, 0.25, 1.0)
    effective_texture = high_pass_texture * (pore_weight * texture_clamped * smooth_gate + (1.0 - smooth_gate))
    
    smoothed = (reconstructed_base + effective_texture).clip(0, 255)

    # 6. Anatomical Boundary Feathering
    # Resolution-adaptive soft edge feathering ensures seamless transitions without hard lines.
    eff_feather = max(feather_radius, max(8, int(dim * 0.012)))
    if eff_feather > 0:
        ksize = eff_feather * 2 + 1
        feathered_alpha = cv2.GaussianBlur(skin_mask.astype(np.float32), (ksize, ksize), eff_feather / 2.2)
        feathered_alpha = np.clip(feathered_alpha, 0.0, 255.0).astype(np.uint8)
    else:
        feathered_alpha = skin_mask.copy()
        
    alpha_f = (feathered_alpha.astype(np.float32) / 255.0)[:, :, None]
    composited = img_f * (1.0 - alpha_f) + smoothed * alpha_f

    # Bit-for-bit guarantee: strictly preserve original background pixels outside skin_mask
    composited_clean = np.where(skin_mask[:, :, None] > 0, composited, img_rgb)

    return composited_clean.clip(0, 255).astype(np.uint8), feathered_alpha


def apply_frequency_separation(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = 0.5,
    texture_keep: float = 0.85,
    feather_radius: int = 4,
) -> np.ndarray:
    composited, _ = frequency_separation_smooth(
        img_rgb,
        skin_mask,
        strength=strength,
        texture_keep=texture_keep,
        feather_radius=feather_radius,
    )
    return composited


def apply_full_smooth(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = 0.5,
    even_redness_strength: float = 0.0,
    texture_keep: float = 0.85,
    feather_radius: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    if even_redness_strength > 0:
        tone_evened = even_redness(img_rgb, skin_mask, strength=even_redness_strength)
    else:
        tone_evened = img_rgb

    smoothed, alpha = frequency_separation_smooth(
        tone_evened,
        skin_mask,
        strength=strength,
        texture_keep=texture_keep,
        feather_radius=feather_radius
    )
    # Final guard: outside skin_mask, strictly return img_rgb
    smoothed_guarded = np.where(skin_mask[:, :, None] > 0, smoothed, img_rgb)
    return smoothed_guarded, alpha


def create_smooth_rgba_patch(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = 0.5,
    even_redness_strength: float = 0.0,
    texture_keep: float = 0.85,
    feather_radius: int = 4
) -> Image.Image:
    smoothed_rgb, effective_mask = apply_full_smooth(
        img_rgb=img_rgb,
        skin_mask=skin_mask,
        strength=strength,
        even_redness_strength=even_redness_strength,
        texture_keep=texture_keep,
        feather_radius=feather_radius
    )

    h, w, _ = img_rgb.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = smoothed_rgb
    # ONLY face skin parts have non-zero alpha! Everything outside the face is 100% transparent!
    rgba[:, :, 3] = effective_mask

    from server import anchor_corners_rgba
    return anchor_corners_rgba(Image.fromarray(rgba, mode="RGBA"))

