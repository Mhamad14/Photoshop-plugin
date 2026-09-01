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
CRITICAL_ANATOMY_CLASSES = {2, 3, 4, 5, 11, 12, 13}

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


def estimate_fitzpatrick_type(base_lab: list) -> str:
    """
    Estimates Fitzpatrick skin phototype (Type I to VI) based on CIE LAB L* luminance.
    Used by downstream modules for tone-adaptive sensitivity and erythema thresholds.
    """
    l_val = base_lab[0]  # OpenCV LAB: 0 to 255
    if l_val > 195:
        return "Type I (Very Fair / Porcelain)"
    elif l_val > 175:
        return "Type II (Fair)"
    elif l_val > 155:
        return "Type III (Medium / Olive)"
    elif l_val > 130:
        return "Type IV (Tan / Moderate Brown)"
    elif l_val > 105:
        return "Type V (Dark Brown)"
    else:
        return "Type VI (Deep / Very Dark)"


def extract_facial_zones(
    class_mask: np.ndarray,
    skin_mask: np.ndarray,
    img_rgb: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Precision Facial Zone Sub-Segmentation:
    Deconstructs facial skin into distinct anatomical regions:
    - Forehead (upper third above eyebrows)
    - Left Cheek & Right Cheek (lateral middle zones)
    - Nose (class 10 nasal ridge & tip)
    - Chin (lower third below mouth)
    - Neck (class 14)
    - Eyes, Brows, Lips (structural features)
    """
    h, w = class_mask.shape[:2]
    zones = {}

    # 1. Structural features
    zones["lips"] = np.isin(class_mask, [12, 13]).astype(np.uint8) * 255
    zones["mouth_interior"] = (class_mask == 11).astype(np.uint8) * 255
    zones["eyes"] = np.isin(class_mask, [4, 5]).astype(np.uint8) * 255
    zones["brows"] = np.isin(class_mask, [2, 3]).astype(np.uint8) * 255
    zones["nose"] = (class_mask == 10).astype(np.uint8) * 255
    zones["neck"] = (class_mask == 14).astype(np.uint8) * 255

    # 2. Geometric landmarks for forehead, cheeks, and chin
    facial_skin = (class_mask == 1).astype(np.uint8) * 255
    
    # Eyebrow center height
    brow_coords = np.argwhere(zones["brows"] > 0)
    eye_coords = np.argwhere(zones["eyes"] > 0)
    lip_coords = np.argwhere(zones["lips"] > 0)

    y_brow = np.min(brow_coords[:, 0]) if len(brow_coords) > 0 else int(h * 0.35)
    y_eye = np.mean(eye_coords[:, 0]) if len(eye_coords) > 0 else int(h * 0.42)
    y_lip_bottom = np.max(lip_coords[:, 0]) if len(lip_coords) > 0 else int(h * 0.72)
    x_center = int(w * 0.5)

    if len(eye_coords) > 0:
        x_left_eye = np.min(eye_coords[:, 1])
        x_right_eye = np.max(eye_coords[:, 1])
    else:
        x_left_eye, x_right_eye = int(w * 0.3), int(w * 0.7)

    # Forehead: skin above eye/brow line
    y_forehead_split = int((y_brow + y_eye) / 2)
    forehead_mask = (facial_skin > 0) & (np.arange(h)[:, None] < y_forehead_split)
    zones["forehead"] = forehead_mask.astype(np.uint8) * 255

    # Chin: skin below lower lip
    chin_mask = (facial_skin > 0) & (np.arange(h)[:, None] > y_lip_bottom)
    zones["chin"] = chin_mask.astype(np.uint8) * 255

    # Cheeks: skin between eye line and chin, lateral to nose center
    mid_face = (facial_skin > 0) & (np.arange(h)[:, None] >= y_forehead_split) & (np.arange(h)[:, None] <= y_lip_bottom)
    
    # Exclude nose region from cheeks
    mid_face[zones["nose"] > 0] = 0

    zones["left_cheek"] = (mid_face & (np.arange(w)[None, :] < x_center)).astype(np.uint8) * 255
    zones["right_cheek"] = (mid_face & (np.arange(w)[None, :] >= x_center)).astype(np.uint8) * 255

    return zones


def segment_face_skin(
    img_rgb: np.ndarray,
    include_neck: bool = True,
    feather_radius: int = 3,
    protect_nostrils: bool = True,
    protect_vermilion: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Layer 1 — Precision Anatomical Skin Segmentation & Facial Zoning.
    
    Extracts high-precision skin_mask with strict anatomical safety exclusion zones:
    1. Vermilion lip border (Cupid's bow & lip contours)
    2. Eyelid crease, lash lines, and tear ducts (canthus)
    3. Eyebrow contours
    4. Nostril cavity / nasal base crevices
    
    Returns:
        skin_mask: uint8 2D array (0 to 255)
        metadata: Dict with detected class statistics, Fitzpatrick type, base tone, and anatomical facial zone masks
    """
    h, w = img_rgb.shape[:2]
    parser = get_bisenet_parser()
    
    if parser is None:
        logger.warning("BiSeNet parser unavailable, falling back to HSV/YCbCr skin detection")
        return fallback_skin_segmentation(img_rgb, feather_radius=feather_radius)
        
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    try:
        class_mask = parser.parse(img_bgr)
    except Exception as e:
        logger.error(f"Error during BiSeNet parse: {e}")
        return fallback_skin_segmentation(img_rgb, feather_radius=feather_radius)
        
    # Target skin classes
    target_classes = set(SKIN_CLASSES)
    if include_neck:
        target_classes.update(NECK_CLASSES)
        
    # Create raw skin mask
    raw_mask = np.isin(class_mask, list(target_classes)).astype(np.uint8) * 255
    
    # -------------------------------------------------------------
    # ANATOMICAL SAFETY EXCLUSION ZONES
    # -------------------------------------------------------------
    # 1. Protect Lip Vermilion Border (Cupid's bow)
    if protect_vermilion:
        lip_region = np.isin(class_mask, [12, 13]).astype(np.uint8) * 255
        if np.any(lip_region):
            k_lip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated_lips = cv2.dilate(lip_region, k_lip)
            raw_mask[dilated_lips > 0] = 0

    # 2. Protect Eye Margins & Eyebrows
    eye_brow_region = np.isin(class_mask, [2, 3, 4, 5]).astype(np.uint8) * 255
    if np.any(eye_brow_region):
        k_eye = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_eyes = cv2.dilate(eye_brow_region, k_eye)
        raw_mask[dilated_eyes > 0] = 0

    # 3. Nostril / dark crevice protection in the nose area (class 10)
    if protect_nostrils:
        nose_region = (class_mask == 10)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        nostril_candidates = (gray < 48) & nose_region
        if np.any(nostril_candidates):
            k_nos = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            nostril_mask = cv2.dilate(nostril_candidates.astype(np.uint8) * 255, k_nos)
            raw_mask[nostril_mask > 0] = 0

    # Morphological cleanup
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Extract sub-facial zones
    zones = extract_facial_zones(class_mask, cleaned_mask, img_rgb)

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

    fitzpatrick = estimate_fitzpatrick_type(base_lab)

    # Soft edge feathering
    if feather_radius > 0:
        ksize = feather_radius * 2 + 1
        final_mask = cv2.GaussianBlur(cleaned_mask, (ksize, ksize), 0)
    else:
        final_mask = cleaned_mask

    # Non-skin anatomy mask (lips, mouth, nostrils, eyes, background)
    non_skin = np.ones((h, w), dtype=np.uint8) * 255
    non_skin[cleaned_mask > 128] = 0

    metadata = {
        "classes_present": [int(c) for c in np.unique(class_mask)],
        "skin_pixel_count": int(np.sum(cleaned_mask > 128)),
        "skin_percentage": float(np.sum(cleaned_mask > 128) / (h * w) * 100.0),
        "base_tone_rgb": base_rgb,
        "base_tone_lab": base_lab,
        "fitzpatrick_type": fitzpatrick,
        "zones_count": len(zones),
        "class_mask": class_mask,
        "non_skin_mask": non_skin,
        "nose_mask": zones.get("nose", np.zeros((h, w), dtype=np.uint8)),
        "lips_mask": zones.get("lips", np.zeros((h, w), dtype=np.uint8))
    }
    
    return final_mask, metadata


def fallback_skin_segmentation(img_rgb: np.ndarray, feather_radius: int = 3) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fallback color-space skin segmenter (YCbCr + HSV)."""
    h, w = img_rgb.shape[:2]
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    cr = img_ycrcb[:, :, 1]
    cb = img_ycrcb[:, :, 2]
    skin_ycrcb = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)

    h_chan = img_hsv[:, :, 0]
    s_chan = img_hsv[:, :, 1]
    skin_hsv = ((h_chan <= 25) | (h_chan >= 170)) & (s_chan >= 20) & (s_chan <= 200)

    raw_mask = (skin_ycrcb & skin_hsv).astype(np.uint8) * 255

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
        base_lab = [170.0, 140.0, 140.0]
    else:
        base_rgb = [210, 170, 150]
        base_lab = [170.0, 140.0, 140.0]

    return final_mask, {
        "classes_present": [],
        "skin_pixel_count": int(np.sum(cleaned_mask > 128)),
        "skin_percentage": float(np.sum(cleaned_mask > 128) / (h * w) * 100.0),
        "base_tone_rgb": base_rgb,
        "base_tone_lab": base_lab,
        "fitzpatrick_type": estimate_fitzpatrick_type(base_lab),
        "zones_count": 0
    }
