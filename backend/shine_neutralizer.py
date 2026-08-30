import logging
import cv2
import numpy as np
from PIL import Image
from typing import Tuple

logger = logging.getLogger("shine_neutralizer")


def neutralize_skin_shine(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = 0.5,
    threshold: float = 0.75,
    feather_radius: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    AI Specular Highlight & Oiliness / Flash Glare Neutralizer.
    
    Detects blown-out specular highlights on forehead, nose, and cheekbones,
    and replaces harsh hot-spots with smooth local matte skin tones.
    
    Args:
        img_rgb: uint8 RGB numpy array (HxWx3)
        skin_mask: uint8 2D skin mask (HxW)
        strength: float between 0.0 and 1.0
        threshold: float 0.5 to 0.95 (luminance percentile threshold for shine)
        feather_radius: blur radius for patch edges
        
    Returns:
        composited_rgb: uint8 HxWx3
        patch_alpha: uint8 HxW
    """
    if strength <= 0:
        return img_rgb.copy(), np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    h, w, _ = img_rgb.shape
    skin_binary = (skin_mask > 30).astype(np.float32)
    if np.sum(skin_binary) < 100:
        return img_rgb.copy(), np.zeros((h, w), dtype=np.uint8)

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # Compute local skin baseline luminance (matte tone)
    sigma = max(12.0, min(h, w) * 0.05)
    norm_w = cv2.GaussianBlur(skin_binary, (0, 0), sigma) + 1e-5
    l_matte = cv2.GaussianBlur(l_chan * skin_binary, (0, 0), sigma) / norm_w
    a_matte = cv2.GaussianBlur(a_chan * skin_binary, (0, 0), sigma) / norm_w
    b_matte = cv2.GaussianBlur(b_chan * skin_binary, (0, 0), sigma) / norm_w

    # Shine spots are pixels where L is substantially higher than local matte skin
    # and chromatic saturation is washed out (towards pure white)
    shine_diff = (l_chan - l_matte) * skin_binary
    shine_threshold = 25.0 * (1.0 - (threshold - 0.5))

    shine_mask = np.clip((shine_diff - shine_threshold) / (shine_threshold + 1e-5), 0.0, 1.0)
    shine_mask = shine_mask * skin_binary

    if np.sum(shine_mask > 0.1) < 10:
        return img_rgb.copy(), np.zeros((h, w), dtype=np.uint8)

    # Extract high-frequency skin pore texture from surrounding healthy skin
    img_f = img_rgb.astype(np.float32)
    blurred_img = cv2.GaussianBlur(img_f, (7, 7), 0)
    high_freq_texture = img_f - blurred_img

    # Soften the shine mask
    ksize = feather_radius * 2 + 1
    shine_soft = cv2.GaussianBlur(shine_mask, (ksize, ksize), 0)

    # Retain 18% organic specular sheen so highlights look natural and alive (not dead/gray)
    l_corrected_matte = l_matte + (l_chan - l_matte) * 0.18
    matte_rgb = cv2.cvtColor(cv2.merge([l_corrected_matte, a_matte, b_matte]).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    
    # Inject healthy skin pore micro-texture into the matte zone
    matte_textured = (matte_rgb + high_freq_texture * 0.65).clip(0, 255)

    # Blend shine pixels toward textured matte tone
    s_factor = (shine_soft * min(1.0, strength) * 0.88)[:, :, None]
    composited = (img_f * (1.0 - s_factor) + matte_textured * s_factor).clip(0, 255).astype(np.uint8)

    patch_alpha = (shine_soft * min(1.0, strength) * 255.0).clip(0, 255).astype(np.uint8)

    return composited, patch_alpha


def create_shine_neutralizer_rgba_patch(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = 0.5,
    threshold: float = 0.75,
    feather_radius: int = 4
) -> Image.Image:
    """Returns transparent RGBA patch neutralizing oily glare."""
    composited, alpha = neutralize_skin_shine(
        img_rgb, skin_mask, strength=strength, threshold=threshold, feather_radius=feather_radius
    )
    r = Image.fromarray(composited[:, :, 0])
    g = Image.fromarray(composited[:, :, 1])
    b = Image.fromarray(composited[:, :, 2])
    a = Image.fromarray(alpha)
    return Image.merge("RGBA", (r, g, b, a))
