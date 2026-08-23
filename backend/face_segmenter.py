import logging
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger("face_segmenter")

# BiSeNet 19 Classes (CelebAMask-HQ mapping)
# 0: background, 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye, 
# 6: eye_g, 7: l_ear, 8: r_ear, 9: ear_r, 10: nose, 11: mouth, 
# 12: u_lip, 13: l_lip, 14: neck, 15: neck_l, 16: cloth, 17: hair, 18: hat

SKIN_CLASSES = {1, 10}  # Facial skin + nose skin
NECK_CLASSES = {14}      # Neck skin (optional)
EXCLUDE_CLASSES = {0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18}

_bisenet_model = None

def get_bisenet_parser():
    global _bisenet_model
    if _bisenet_model is None:
        try:
            from uniface.parsing import BiSeNet
            logger.info("Initializing BiSeNet Face Parser...")
            _bisenet_model = BiSeNet()
            logger.info("BiSeNet Face Parser ready.")
        except Exception as e:
            logger.error(f"Failed to load BiSeNet: {e}", exc_info=True)
            _bisenet_model = None
    return _bisenet_model


def segment_face_skin(
    img_rgb: np.ndarray,
    include_neck: bool = True,
    feather_radius: int = 3,
    protect_nostrils: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Layer 1 — Skin/Face Segmentation.
    Extracts high-precision skin_mask separating skin from hair, eyes, brows, lips, and nostrils.
    
    Args:
        img_rgb: uint8 RGB numpy array
        include_neck: whether to include neck in the skin mask
        feather_radius: radius for soft edge boundary
        protect_nostrils: exclude dark nostril holes via intensity thresholding inside nose region
        
    Returns:
        skin_mask: uint8 2D array (0 to 255)
        metadata: Dict with detected class statistics and sampled base skin tone
    """
    h, w = img_rgb.shape[:2]
    parser = get_bisenet_parser()
    
    if parser is None:
        logger.warning("BiSeNet parser unavailable, falling back to HSV/YCbCr skin detection")
        return fallback_skin_segmentation(img_rgb, feather_radius=feather_radius)
        
    # uniface BiSeNet expects BGR image
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    try:
        class_mask = parser.parse(img_bgr)
    except Exception as e:
        logger.error(f"Error during BiSeNet parse: {e}")
        return fallback_skin_segmentation(img_rgb, feather_radius=feather_radius)
        
    # Build target classes
    target_classes = set(SKIN_CLASSES)
    if include_neck:
        target_classes.update(NECK_CLASSES)
        
    # Create binary mask
    raw_mask = np.isin(class_mask, list(target_classes)).astype(np.uint8) * 255
    
    # Nostril / dark crevice protection in the nose area (class 10)
    if protect_nostrils:
        nose_region = (class_mask == 10)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        # Nostrils are typically very dark spots inside the nose area
        nostril_candidates = (gray < 45) & nose_region
        # Dilate slightly to protect nostril boundaries
        if np.any(nostril_candidates):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            nostril_mask = cv2.dilate(nostril_candidates.astype(np.uint8) * 255, k)
            raw_mask[nostril_mask > 0] = 0

    # Clean up small holes in the skin mask
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Sample healthy base skin tone (median RGB and LAB of skin pixels)
    skin_pixels = img_rgb[cleaned_mask > 128]
    if len(skin_pixels) > 0:
        base_rgb = np.median(skin_pixels, axis=0).astype(int).tolist()
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        skin_lab_pixels = img_lab[cleaned_mask > 128]
        base_lab = np.median(skin_lab_pixels, axis=0).astype(float).tolist()
    else:
        base_rgb = [210, 170, 150]
        base_lab = [170.0, 140.0, 140.0]

    # Apply soft edge feathering
    if feather_radius > 0:
        ksize = feather_radius * 2 + 1
        final_mask = cv2.GaussianBlur(cleaned_mask, (ksize, ksize), 0)
    else:
        final_mask = cleaned_mask

    metadata = {
        "classes_present": [int(c) for c in np.unique(class_mask)],
        "skin_pixel_count": int(np.sum(cleaned_mask > 128)),
        "skin_percentage": float(np.sum(cleaned_mask > 128) / (h * w) * 100.0),
        "base_tone_rgb": base_rgb,
        "base_tone_lab": base_lab
    }
    
    return final_mask, metadata


def fallback_skin_segmentation(img_rgb: np.ndarray, feather_radius: int = 3) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fallback color-space skin segmenter (YCbCr + HSV) in case BiSeNet is not loaded.
    """
    h, w = img_rgb.shape[:2]
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # YCrCb skin thresholds
    cr = img_ycrcb[:, :, 1]
    cb = img_ycrcb[:, :, 2]
    skin_ycrcb = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)

    # HSV skin thresholds
    h_chan = img_hsv[:, :, 0]
    s_chan = img_hsv[:, :, 1]
    skin_hsv = ((h_chan <= 25) | (h_chan >= 170)) & (s_chan >= 20) & (s_chan <= 200)

    raw_mask = (skin_ycrcb & skin_hsv).astype(np.uint8) * 255

    # Morphology to fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

    if feather_radius > 0:
        ksize = feather_radius * 2 + 1
        final_mask = cv2.GaussianBlur(cleaned_mask, (ksize, ksize), 0)
    else:
        final_mask = cleaned_mask

    skin_pixels = img_rgb[cleaned_mask > 128]
    if len(skin_pixels) > 0:
        base_rgb = np.median(skin_pixels, axis=0).astype(int).tolist()
    else:
        base_rgb = [210, 170, 150]

    return final_mask, {
        "classes_present": [],
        "skin_pixel_count": int(np.sum(cleaned_mask > 128)),
        "skin_percentage": float(np.sum(cleaned_mask > 128) / (h * w) * 100.0),
        "base_tone_rgb": base_rgb,
        "base_tone_lab": [170.0, 140.0, 140.0]
    }
