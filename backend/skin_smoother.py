import logging
from typing import Optional, List

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
    texture_keep: float = 0.4,
    feather_radius: int = 4,
) -> tuple:
    """
    Professional frequency-separation skin smoothing, confined to the skin mask.

    Splits the image into:
      - base layer: edge-aware (bilateral) low-frequency color/tone
      - detail layer: high-frequency pores/texture/noise

    Inside the skin mask the detail layer is attenuated (smoothing out blemishes,
    blackheads, fine lines and rough texture) while the base layer keeps natural
    tone transitions and edges (eyes, lips, hairline stay sharp).

    Returns:
        smoothed_composited_rgb: uint8 HxWx3 (original outside the mask)
        feathered_alpha: uint8 HxW alpha for non-destructive patch placement
    """
    h, w, _ = img_rgb.shape
    mask_binary = (skin_mask > 30).astype(np.uint8)
    if np.sum(mask_binary) < 100:
        return img_rgb.copy(), skin_mask.copy()

    img_f = img_rgb.astype(np.float32)
    dim = min(h, w)
    
    # Scale bilateral parameters dynamically based on portrait resolution
    d_val = max(5, min(15, int(dim * 0.006) | 1))
    sigma_space = max(5.0, min(25.0, dim * 0.008))
    sigma_color = max(25.0, min(55.0, 30.0 + strength * 20.0))

    # Edge-preserving base layer (bilateral filter preserves facial features, eyes, lips, jawline)
    base = cv2.bilateralFilter(img_rgb, d=d_val, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    if strength >= 0.45:
        base = cv2.bilateralFilter(base, d=d_val, sigmaColor=sigma_color * 0.8, sigmaSpace=sigma_space * 0.8)
    base_f = base.astype(np.float32)

    detail = img_f - base_f

    # High-frequency detail attenuation with skin pore texture recovery
    strength = max(0.0, min(1.0, strength))
    texture_keep = max(0.05, min(1.0, texture_keep))
    detail_scale = 1.0 - strength * (1.0 - texture_keep)

    smoothed = base_f + (detail * detail_scale)
    
    # Add subtle micro-pore preservation boost for high-end magazine finish
    if texture_keep > 0.2:
        gray_detail = cv2.cvtColor(np.abs(detail).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        pore_mask = ((gray_detail > 3) & (gray_detail < 25)).astype(np.float32)[:, :, None]
        smoothed = smoothed + (detail * pore_mask * (texture_keep * 0.25))

    smoothed = smoothed.clip(0, 255)

    # Confine to skin region with soft feathered blending
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
    texture_keep: float = 0.4,
    feather_radius: int = 4,
) -> tuple:
    """
    Complete smooth pipeline: even redness first, then frequency separation.
    Returns (composited_rgb, feathered_alpha).
    """
    evened = even_redness(img_rgb, skin_mask, strength=max(0.0, min(1.0, strength)))
    return frequency_separation_smooth(
        img_rgb=evened,
        skin_mask=skin_mask,
        strength=strength,
        texture_keep=texture_keep,
        feather_radius=feather_radius,
    )


def create_smooth_rgba_patch(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = 0.5,
    texture_keep: float = 0.4,
    feather_radius: int = 4,
) -> Image.Image:
    """Transparent RGBA PNG patch containing only the smoothed skin region."""
    composited, alpha = apply_full_smooth(
        img_rgb=img_rgb,
        skin_mask=skin_mask,
        strength=strength,
        texture_keep=texture_keep,
        feather_radius=feather_radius,
    )
    r = Image.fromarray(composited[:, :, 0])
    g = Image.fromarray(composited[:, :, 1])
    b = Image.fromarray(composited[:, :, 2])
    a = Image.fromarray(alpha)
    return Image.merge("RGBA", (r, g, b, a))
