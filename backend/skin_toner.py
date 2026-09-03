import logging
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Optional, List

logger = logging.getLogger("skin_toner")


def calculate_tone_lift(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = 0.35,
    base_tone_lab: Optional[List[float]] = None,
    feather_radius: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Layer 5 — Skin Lightening.
    Applies a natural, tone-relative brightening boost to facial skin within skin_mask.
    
    Args:
        img_rgb: uint8 RGB image
        skin_mask: uint8 2D mask (0 to 255)
        strength: lightening intensity from 0.0 (no change) to 1.0 (strong)
        base_tone_lab: [L, a, b] baseline skin tone sampled from user/detector
        feather_radius: soft edge transition radius in pixels
        
    Returns:
        lightened_rgb: RGB image with brightened skin
        feathered_alpha: uint8 2D alpha channel (0 to 255)
    """
    h, w, _ = img_rgb.shape
    mask_binary = (skin_mask > 30).astype(np.uint8)
    
    if np.sum(mask_binary) == 0:
        return img_rgb.copy(), np.zeros((h, w), dtype=np.uint8)

    # Convert to CIE LAB space
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan, a_chan, b_chan = cv2.split(img_lab)

    # Determine baseline lightness L0
    if base_tone_lab is not None and len(base_tone_lab) >= 1:
        base_l = float(base_tone_lab[0])
    else:
        # Compute median lightness over skin mask
        skin_l = l_chan[mask_binary > 0]
        base_l = float(np.median(skin_l)) if len(skin_l) > 0 else 160.0

    # Calculate smooth midtone lift
    # Maximum boost in midtones, tapering off in deep shadows and bright highlights to prevent clipping
    # L ranges from 0 to 255 in OpenCV LAB
    norm_l = l_chan / 255.0
    # Bell curve weighting for natural photographic midtone enhancement
    midtone_weight = 4.0 * norm_l * (1.0 - norm_l)
    
    # Scale factor based on strength and base skin tone
    # Darker skin tones get a rich luminous glow without chalkiness; lighter tones get porcelain clarity
    tone_scale = max(0.6, min(1.4, (255.0 - base_l) / 128.0))
    lift_amount = strength * 34.0 * tone_scale * midtone_weight

    new_l = np.clip(l_chan + lift_amount, 0, 255)

    # Tone-Adaptive Chroma Radiance (prevents ashy/chalky or jaundiced casts)
    if base_l > 175:
        # Fair/Porcelain skin: slight rose-alabaster warmth
        chroma_a_boost = strength * 1.2 * midtone_weight
        chroma_b_boost = strength * 0.6 * midtone_weight
    elif base_l > 130:
        # Medium/Olive skin: balanced golden radiance
        chroma_a_boost = strength * 0.8 * midtone_weight
        chroma_b_boost = strength * 1.4 * midtone_weight
    else:
        # Deep/Dark skin: rich warm bronze radiance
        chroma_a_boost = strength * 1.5 * midtone_weight
        chroma_b_boost = strength * 2.2 * midtone_weight

    new_a = np.clip(a_chan + chroma_a_boost, 0, 255)
    new_b = np.clip(b_chan + chroma_b_boost, 0, 255)

    merged_lab = cv2.merge([new_l, new_a, new_b]).astype(np.uint8)
    lightened_full_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

    # Feather the skin mask for seamless alpha blending
    if feather_radius > 0:
        ksize = feather_radius * 2 + 1
        feathered_alpha = cv2.GaussianBlur(skin_mask, (ksize, ksize), 0)
    else:
        feathered_alpha = skin_mask.copy()

    return lightened_full_rgb, feathered_alpha


def apply_relative_lighten(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = 0.35,
    base_tone_lab: Optional[List[float]] = None,
    feather_radius: int = 4,
) -> np.ndarray:
    lightened_rgb, _ = calculate_tone_lift(
        img_rgb=img_rgb,
        skin_mask=skin_mask,
        strength=strength,
        base_tone_lab=base_tone_lab,
        feather_radius=feather_radius,
    )
    return lightened_rgb


def create_lightened_rgba_patch(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = 0.35,
    base_tone_lab: Optional[List[float]] = None,
    feather_radius: int = 4
) -> Image.Image:
    """
    Creates a transparent RGBA PNG patch containing only the lightened skin region.
    Ready for non-destructive placement on top of the Photoshop document.
    """
    lightened_rgb, alpha = calculate_tone_lift(
        img_rgb=img_rgb,
        skin_mask=skin_mask,
        strength=strength,
        base_tone_lab=base_tone_lab,
        feather_radius=feather_radius
    )

    r = Image.fromarray(lightened_rgb[:, :, 0])
    g = Image.fromarray(lightened_rgb[:, :, 1])
    b = Image.fromarray(lightened_rgb[:, :, 2])
    a = Image.fromarray(alpha)

    return Image.merge("RGBA", (r, g, b, a))
