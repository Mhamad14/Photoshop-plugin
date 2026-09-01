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
    norm_w = cv2.GaussianBlur(valid_weight, (0, 0), sigma)
    blurred_a = cv2.GaussianBlur(a_chan * valid_weight, (0, 0), sigma)
    blurred_b = cv2.GaussianBlur(b_chan * valid_weight, (0, 0), sigma)

    a_base = np.where(norm_w > 1e-3, blurred_a / (norm_w + 1e-6), a_chan)
    b_base = np.where(norm_w > 1e-3, blurred_b / (norm_w + 1e-6), b_chan)

    a_base = np.clip(a_base, 134.0, 168.0)
    b_base = np.clip(b_base, 132.0, 175.0)

    excess_a = np.maximum(0.0, a_chan - (a_base + 2.5))
    
    blush_protect = np.clip((a_chan - 156.0) / 24.0, 0.0, 1.0)
    support = np.clip(norm_w / 0.12, 0.0, 1.0)
    redness_factor = (strength * (1.0 - blush_protect * 0.65) * skin_w * support)

    a_new = a_chan - (excess_a * redness_factor * 0.70)
    b_new = b_chan * (1.0 - redness_factor * 0.10) + b_base * (redness_factor * 0.10)

    a_new = np.maximum(a_new, 132.0)
    b_new = np.maximum(b_new, 130.0)

    merged = cv2.merge([l_chan, a_new, b_new]).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


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

    # 1. Base Low-Frequency Layer with boundary skin normalization
    skin_f_weight = (mask_binary > 0).astype(np.float32)
    norm_low = cv2.GaussianBlur(skin_f_weight, (0, 0), sigma_low) + 1e-5
    base_low = cv2.GaussianBlur(img_f * skin_f_weight[:, :, None], (0, 0), sigma_low) / norm_low[:, :, None]
    base_low = np.where(skin_f_weight[:, :, None] > 0.01, base_low, img_f)
    
    # 2. Intermediate Smooth Layer (edge-preserving guided bilateral filter)
    d_val = max(5, min(13, int(dim * 0.005) | 1))
    sigma_space = max(6.0, min(24.0, dim * 0.008))
    sigma_color = max(20.0, min(50.0, 25.0 + strength * 20.0))
    
    base_mid = cv2.bilateralFilter(img_rgb, d=d_val, sigmaColor=sigma_color, sigmaSpace=sigma_space).astype(np.float32)
    if strength > 0.5:
        base_mid = cv2.bilateralFilter(base_mid.clip(0, 255).astype(np.uint8), d=d_val, sigmaColor=sigma_color * 0.8, sigmaSpace=sigma_space * 0.8).astype(np.float32)

    # 3. High-Frequency Micro-Pore Texture Band
    high_pass_texture = img_f - base_mid

    # 4. Mid-Frequency Roughness Band
    mid_roughness = base_mid - base_low

    # 5. Modulate Smoothing: Smooth mid-roughness while protecting high-pass pores
    strength_clamped = max(0.0, min(1.0, strength))
    texture_clamped = max(0.10, min(1.0, texture_keep))

    smoothed_mid = mid_roughness * (1.0 - strength_clamped * 0.75)
    reconstructed_base = base_low + smoothed_mid

    # Pore-contrast booster
    gray_texture = cv2.cvtColor(np.abs(high_pass_texture).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    pore_envelope = ((gray_texture >= 2) & (gray_texture <= 32)).astype(np.float32)[:, :, None]
    effective_texture = high_pass_texture * (texture_clamped + (pore_envelope * 0.20 * texture_clamped))
    
    smoothed = reconstructed_base + effective_texture
    smoothed = smoothed.clip(0, 255)

    # 6. Structural Feature Edge Protection (Prevents any blur on nose contour, nostrils, lips, eyes)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    edge_protection = np.clip((grad_mag - 16.0) / 28.0, 0.0, 1.0)[:, :, None]
    
    # Blend smoothed result with original image on structural edges
    preserved_smoothed = smoothed * (1.0 - edge_protection * 0.92) + img_f * (edge_protection * 0.92)

    # 7. Soft Feathered Blending within facial skin mask
    if feather_radius > 0:
        ksize = feather_radius * 2 + 1
        feathered_alpha = cv2.GaussianBlur(skin_mask, (ksize, ksize), 0)
    else:
        feathered_alpha = skin_mask.copy()
        
    alpha_f = (feathered_alpha.astype(np.float32) / 255.0)[:, :, None]
    composited = img_f * (1.0 - alpha_f) + preserved_smoothed * alpha_f

    return composited.clip(0, 255).astype(np.uint8), feathered_alpha


def apply_full_smooth(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = 0.5,
    even_redness_strength: float = 0.4,
    texture_keep: float = 0.85,
    feather_radius: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    if even_redness_strength > 0:
        tone_evened = even_redness(img_rgb, skin_mask, strength=even_redness_strength)
    else:
        tone_evened = img_rgb

    return frequency_separation_smooth(
        tone_evened,
        skin_mask,
        strength=strength,
        texture_keep=texture_keep,
        feather_radius=feather_radius
    )


def create_smooth_rgba_patch(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = 0.5,
    even_redness_strength: float = 0.4,
    texture_keep: float = 0.85,
    feather_radius: int = 4
) -> Tuple[np.ndarray, Dict[str, Any]]:
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
    rgba[:, :, 3] = effective_mask

    meta = {
        "strength": strength,
        "texture_keep": texture_keep,
        "even_redness_strength": even_redness_strength,
        "feather_radius": feather_radius,
        "layer_name": f"AI Skin Smoothing ({int(strength * 100)}%)"
    }

    return rgba, meta
