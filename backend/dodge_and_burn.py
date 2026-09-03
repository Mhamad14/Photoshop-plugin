import logging
import cv2
import numpy as np
from PIL import Image
from typing import Tuple

logger = logging.getLogger("dodge_and_burn")


def generate_dodge_and_burn_map(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = 0.5,
    softness: float = 0.6,
    feather_radius: int = 4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Studio-Grade Dual-Scale AI Dodge & Burn (D&B) Engine.
    
    Principles:
    1. Micro-Contrast Isolation: Targets ONLY patchy micro-shadows & blemish depressions (2px - 14px).
    2. 3D Facial Volume Preservation: Zeroes out broad macro-gradients (>30px) so cheekbone highlights,
       nose bridge lighting, and jawline shadows remain 100% natural.
    3. 100% Organic Pore Texture Retention: Modulates only the low/mid frequency luminance band without
       touching the high-frequency pore structure.
       
    Args:
        img_rgb: uint8 RGB numpy array (HxWx3)
        skin_mask: uint8 2D skin mask (HxW)
        strength: float between 0.0 and 1.0 (D&B intensity)
        softness: float between 0.1 and 1.0 (tonal blending scale)
        feather_radius: soft border blending
        
    Returns:
        composited_rgb: uint8 HxWx3 result
        db_gray_map: uint8 HxW (50% neutral 128 mid-gray overlay map for Photoshop Soft-Light blend)
        feathered_alpha: uint8 HxW alpha mask
    """
    if strength <= 0.0:
        return img_rgb.copy(), np.full(img_rgb.shape[:2], 128, dtype=np.uint8), skin_mask.copy()

    h, w, _ = img_rgb.shape
    skin_binary = (skin_mask > 25).astype(np.float32)
    if np.sum(skin_binary) < 100:
        return img_rgb.copy(), np.full((h, w), 128, dtype=np.uint8), skin_mask.copy()

    # Convert to LAB to isolate pure Luminance (L*)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # -------------------------------------------------------------
    # Dual-Scale Bandpass Decomposition
    # -------------------------------------------------------------
    # Macro scale (broad 3D facial lighting - cheekbones, jaw, nose contour)
    sigma_macro = max(18.0, min(h, w) * (0.05 * softness + 0.03))
    norm_macro = cv2.GaussianBlur(skin_binary, (0, 0), sigma_macro) + 1e-5
    l_macro = cv2.GaussianBlur(l_chan * skin_binary, (0, 0), sigma_macro) / norm_macro

    # Micro scale (patchy blemish shadows, blotchy tone transitions)
    sigma_micro = max(3.5, sigma_macro * 0.25)
    norm_micro = cv2.GaussianBlur(skin_binary, (0, 0), sigma_micro) + 1e-5
    l_micro = cv2.GaussianBlur(l_chan * skin_binary, (0, 0), sigma_micro) / norm_micro

    # The difference isolates ONLY micro-luminance discrepancies
    # lum_diff > 0: micro-shadow (needs dodging / lightening)
    # lum_diff < 0: micro-hotspot (needs burning / darkening)
    lum_diff = (l_macro - l_micro)

    # -------------------------------------------------------------
    # Anatomical & Luminance Safety Clamping
    # -------------------------------------------------------------
    # Prevent over-dodging in natural deep structural shadows (under chin, nostril base)
    shadow_protect = np.clip((l_chan - 30.0) / 35.0, 0.0, 1.0)
    
    # Prevent over-burning in natural specular highlights (cheekbone apex)
    highlight_protect = np.clip((242.0 - l_chan) / 28.0, 0.0, 1.0)

    # Edge Gradient Protection: protect anatomical lines (eyelids, lips, nostril creases)
    grad_gray = cv2.morphologyEx(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    edge_protect = np.clip(1.0 - (grad_gray.astype(np.float32) / 45.0), 0.15, 1.0)

    # Scale correction by user strength and safety envelopes
    strength_factor = max(0.0, min(1.0, strength)) * 0.65
    
    correction = lum_diff * strength_factor * skin_binary * shadow_protect * highlight_protect * edge_protect

    # Apply luminance adjustment
    l_corrected = (l_chan + correction).clip(0, 255)

    # Generate 50% neutral gray map for Photoshop Soft-Light / Overlay blend
    # 128 = 50% Gray (neutral). > 128 dodges, < 128 burns
    db_gray = (128.0 + correction * (128.0 / 30.0)).clip(0, 255).astype(np.uint8)

    # Reconstruct RGB
    merged_lab = cv2.merge([l_corrected, a_chan, b_chan]).astype(np.uint8)
    corrected_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

    # Soft alpha blend
    if feather_radius > 0:
        ksize = feather_radius * 2 + 1
        feathered_alpha = cv2.GaussianBlur(skin_mask, (ksize, ksize), 0)
    else:
        feathered_alpha = skin_mask.copy()

    alpha_f = (feathered_alpha.astype(np.float32) / 255.0)[:, :, None]
    composited_rgb = (img_rgb.astype(np.float32) * (1.0 - alpha_f) + corrected_rgb.astype(np.float32) * alpha_f).clip(0, 255).astype(np.uint8)

    return composited_rgb, db_gray, feathered_alpha


def apply_ai_dodge_and_burn(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = 0.5,
    softness: float = 0.6,
    feather_radius: int = 4,
) -> np.ndarray:
    composited_rgb, _, _ = generate_dodge_and_burn_map(
        img_rgb, skin_mask, strength=strength, softness=softness, feather_radius=feather_radius
    )
    return composited_rgb


def create_dodge_and_burn_rgba_patch(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = 0.5,
    softness: float = 0.6,
    feather_radius: int = 4
) -> Image.Image:
    """Returns transparent RGBA PNG patch of the Dodge & Burn layer."""
    composited_rgb, _, alpha = generate_dodge_and_burn_map(
        img_rgb, skin_mask, strength=strength, softness=softness, feather_radius=feather_radius
    )
    r = Image.fromarray(composited_rgb[:, :, 0])
    g = Image.fromarray(composited_rgb[:, :, 1])
    b = Image.fromarray(composited_rgb[:, :, 2])
    a = Image.fromarray(alpha)
    return Image.merge("RGBA", (r, g, b, a))
