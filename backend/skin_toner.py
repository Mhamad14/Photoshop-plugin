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

    # Determine baseline lightness L0 from actual skin (ignoring dark beard/shadows < 80)
    if base_tone_lab is not None and len(base_tone_lab) >= 1 and float(base_tone_lab[0]) > 100:
        base_l = float(base_tone_lab[0])
    else:
        valid_skin_l = l_chan[(mask_binary > 0) & (l_chan > 80.0)]
        base_l = float(np.median(valid_skin_l)) if len(valid_skin_l) > 0 else 160.0

    # Calculate smooth midtone lift
    norm_l = l_chan / 255.0
    midtone_weight = 3.5 * norm_l * (1.0 - norm_l) ** 0.8
    
    # Hair Protection Gate:
    # Dark facial hair (beard, mustache, eyebrows with L* < 60) receives 0 boost,
    # completely preventing facial hair from turning gray, chalky, or washed out.
    hair_gate = np.clip((l_chan - 58.0) / 25.0, 0.0, 1.0)

    tone_scale = max(0.75, min(1.35, (255.0 - base_l) / 110.0))
    lift_amount = strength * 46.0 * tone_scale * midtone_weight * hair_gate

    new_l = np.where(mask_binary > 0, np.clip(l_chan + lift_amount, 0, 255), l_chan)

    # Tone-Adaptive Warm Golden & Peachy Radiance (Studio Portrait Glow)
    # Proportionally boosts rich warmth and rosy vitality as lightness lifts,
    # completely eliminating washed-out, chalky, or gray cast.
    if base_l > 165:
        # Fair/Porcelain skin: luminous rose-alabaster warmth
        chroma_a_boost = strength * 5.5 * midtone_weight * hair_gate
        chroma_b_boost = strength * 11.5 * midtone_weight * hair_gate
    else:
        # Medium/Olive skin: rich golden-peach radiance
        chroma_a_boost = strength * 6.2 * midtone_weight * hair_gate
        chroma_b_boost = strength * 14.5 * midtone_weight * hair_gate

    new_a = np.where(mask_binary > 0, np.clip(a_chan + chroma_a_boost, 0, 255), a_chan)
    new_b = np.where(mask_binary > 0, np.clip(b_chan + chroma_b_boost, 0, 255), b_chan)

    merged_lab = cv2.merge([new_l, new_a, new_b]).astype(np.uint8)
    lightened_full_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

    # Resolution-adaptive soft edge feathering for seamless blending
    dim = min(h, w)
    eff_feather = max(feather_radius, max(12, int(dim * 0.015)))
    if eff_feather > 0:
        ksize = eff_feather * 2 + 1
        feathered_alpha = cv2.GaussianBlur(skin_mask.astype(np.float32), (ksize, ksize), eff_feather / 2.2)
        feathered_alpha = np.clip(feathered_alpha, 0.0, 255.0).astype(np.uint8)
    else:
        feathered_alpha = skin_mask.copy()

    alpha_f = (feathered_alpha.astype(np.float32) / 255.0)[:, :, None]
    composited_rgb = (img_rgb.astype(np.float32) * (1.0 - alpha_f) + lightened_full_rgb.astype(np.float32) * alpha_f).clip(0, 255).astype(np.uint8)
    composited_rgb = np.where(skin_mask[:, :, None] > 0, composited_rgb, img_rgb)

    return composited_rgb, feathered_alpha



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
    # ONLY face skin parts have non-zero alpha! Outside the face is 100% transparent!
    a = Image.fromarray(alpha)

    patch = Image.merge("RGBA", (r, g, b, a))
    from server import anchor_corners_rgba
    return anchor_corners_rgba(patch)
