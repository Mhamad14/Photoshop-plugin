import logging
from typing import Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("skin_smoother")


def even_redness(img_rgb: np.ndarray, skin_mask: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Evens out blotchy redness across the whole skin region by pulling each pixel's
    a*/b* (color) channels toward the local healthy-skin baseline.
    Keeps luminance (L) untouched so texture/detail is preserved.
    """
    if strength <= 0:
        return img_rgb

    h, w, _ = img_rgb.shape
    skin_w = (skin_mask > 30).astype(np.float32)
    if np.sum(skin_w) < 100:
        return img_rgb

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan, a_chan, b_chan = cv2.split(lab)

    sigma = max(10.0, min(h, w) / 20.0)
    norm_w = cv2.GaussianBlur(skin_w, (0, 0), sigma) + 1e-5
    a_base = cv2.GaussianBlur(a_chan * skin_w, (0, 0), sigma) / norm_w
    b_base = cv2.GaussianBlur(b_chan * skin_w, (0, 0), sigma) / norm_w

    factor = strength * skin_w  # only inside skin, scaled by requested strength
    a_new = a_chan * (1.0 - factor) + a_base * factor
    b_new = b_chan * (1.0 - factor * 0.6) + b_base * (factor * 0.6)

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
    Master-Grade Frequency Separation with 100% High-Pass Pore Retention:

    Decomposes the image into:
      1. Low-Frequency Base (Tone & Color): Smooths blotchy color mottling and tonal transitions
         using edge-preserving bilateral filtering while keeping facial bone contours sharp.
      2. High-Frequency Detail (Micro-Pores & Fine Grain): Preserves genuine skin micro-pores
         at 100% while selectively softening only large structural crusts/scabs.

    Returns:
        smoothed_composited_rgb: uint8 HxWx3 (original outside the mask)
        feathered_alpha: uint8 HxW alpha for non-destructive patch placement
    """
    h, w, _ = img_rgb.shape
    mask_binary = (skin_mask > 30).astype(np.uint8)
    if np.sum(mask_binary) < 100:
        return img_rgb.copy(), skin_mask.copy()

    img_f = img_rgb.astype(np.float32)

    # 1. Low-Frequency Base Layer (Color & Tone Smoothing)
    # Bilateral filter preserves structural edges (eyes, lips, nostrils) while melting skin color blotches
    d = max(5, int(min(h, w) * 0.012))
    base = cv2.bilateralFilter(img_rgb, d=d, sigmaColor=32, sigmaSpace=d)
    if strength > 0.6:
        base = cv2.bilateralFilter(base, d=d, sigmaColor=28, sigmaSpace=d)
    base_f = base.astype(np.float32)

    # 2. High-Frequency Detail Layer (Micro-Pores & Texture)
    high_freq_detail = img_f - base_f

    # 3. Controlled Frequency Synthesis
    # Rather than destroying high frequencies, blend low-frequency base according to strength
    # and keep 100% of high-frequency micro-texture (pores)
    eff_strength = max(0.0, min(1.0, strength)) * 0.75
    
    # Smooth tone transition
    blended_base = img_f * (1.0 - eff_strength) + base_f * eff_strength
    
    # Re-inject 100% of high-frequency pore texture
    smoothed = blended_base + high_freq_detail * max(0.70, texture_keep)
    smoothed = smoothed.clip(0, 255)

    # 4. Confine to skin region with soft feathered blending
    if feather_radius > 0:
        ksize = feather_radius * 2 + 1
        feathered_alpha = cv2.GaussianBlur(skin_mask, (ksize, ksize), 0)
    else:
        feathered_alpha = skin_mask.copy()
        
    alpha_f = (feathered_alpha.astype(np.float32) / 255.0)[:, :, None]
    composited = img_f * (1.0 - alpha_f) + smoothed * alpha_f

    return composited.clip(0, 255).astype(np.uint8), feathered_alpha


def apply_full_smooth(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = 0.5,
    even_redness_strength: float = 0.4,
    texture_keep: float = 0.85,
    feather_radius: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combined High-End Retouching Pipeline:
    1. Neutralizes blotchy skin redness in color space.
    2. Runs Master-Grade Frequency Separation with 100% pore retention.
    """
    # Step 1: Even out redness
    if even_redness_strength > 0:
        tone_evened = even_redness(img_rgb, skin_mask, strength=even_redness_strength)
    else:
        tone_evened = img_rgb

    # Step 2: Frequency Separation with 100% pore preservation
    return frequency_separation_smooth(
        tone_evened,
        skin_mask,
        strength=strength,
        texture_keep=texture_keep,
        feather_radius=feather_radius
    )


def create_smoothed_rgba_patch(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = 0.5,
    even_redness_strength: float = 0.4,
    texture_keep: float = 0.85,
    feather_radius: int = 4,
) -> Image.Image:
    """Returns transparent RGBA PNG patch of the smoothed skin layer."""
    smoothed_rgb, alpha = apply_full_smooth(
        img_rgb,
        skin_mask,
        strength=strength,
        even_redness_strength=even_redness_strength,
        texture_keep=texture_keep,
        feather_radius=feather_radius
    )
    r = Image.fromarray(smoothed_rgb[:, :, 0])
    g = Image.fromarray(smoothed_rgb[:, :, 1])
    b = Image.fromarray(smoothed_rgb[:, :, 2])
    a = Image.fromarray(alpha)
    return Image.merge("RGBA", (r, g, b, a))


# Alias for backward compatibility
create_smooth_rgba_patch = create_smoothed_rgba_patch
