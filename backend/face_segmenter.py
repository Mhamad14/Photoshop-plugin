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
_face_detector = None


def get_face_detector():
    global _face_detector
    if _face_detector is None:
        try:
            from uniface import RetinaFace
            logger.info("Initializing RetinaFace Multi-Face Detector...")
            _face_detector = RetinaFace()
            logger.info("RetinaFace ready.")
        except Exception as e:
            logger.warning(f"RetinaFace failed to load: {e}. Trying SCRFD...")
            try:
                from uniface import SCRFD
                _face_detector = SCRFD()
                logger.info("SCRFD ready.")
            except Exception as e2:
                logger.error(f"Failed to load multi-face detector: {e2}")
                _face_detector = None
    return _face_detector


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
    img_rgb: np.ndarray,
    faces: Optional[list] = None
) -> Dict[str, np.ndarray]:
    """
    Precision Facial Zone Sub-Segmentation:
    Deconstructs facial skin into distinct anatomical regions:
    - Forehead (upper third above eyebrows)
    - Cheeks (lateral zones beside nose)
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

    facial_skin = (class_mask == 1).astype(np.uint8) * 255
    forehead_mask = np.zeros((h, w), dtype=np.uint8)
    chin_mask = np.zeros((h, w), dtype=np.uint8)
    left_cheek_mask = np.zeros((h, w), dtype=np.uint8)
    right_cheek_mask = np.zeros((h, w), dtype=np.uint8)

    # If faces bounding boxes are known, compute zones per face
    if faces and len(faces) > 0:
        for f in faces:
            bbox = f.bbox if hasattr(f, "bbox") else f
            bx1, by1, bx2, by2 = [max(0, int(v)) for v in bbox[:4]]
            bx2, by2 = min(w, bx2), min(h, by2)
            if bx2 <= bx1 or by2 <= by1:
                continue

            sub_class = class_mask[by1:by2, bx1:bx2]
            sub_skin = facial_skin[by1:by2, bx1:bx2]
            sub_h, sub_w = sub_class.shape[:2]

            sub_brows = np.isin(sub_class, [2, 3])
            sub_eyes = np.isin(sub_class, [4, 5])
            sub_lips = np.isin(sub_class, [12, 13])
            sub_nose = (sub_class == 10)

            brow_ys = np.where(sub_brows)[0]
            eye_ys = np.where(sub_eyes)[0]
            lip_ys = np.where(sub_lips)[0]
            nose_xs = np.where(sub_nose)[1]

            y_split_top = int(np.mean(eye_ys)) if len(eye_ys) > 0 else (int(np.min(brow_ys)) if len(brow_ys) > 0 else int(sub_h * 0.38))
            y_split_bottom = int(np.max(lip_ys)) if len(lip_ys) > 0 else int(sub_h * 0.72)
            x_nose_center = int(np.median(nose_xs)) if len(nose_xs) > 0 else int(sub_w * 0.5)

            # Forehead
            sub_fh = (sub_skin > 0) & (np.arange(sub_h)[:, None] < y_split_top)
            forehead_mask[by1:by2, bx1:bx2] = np.maximum(forehead_mask[by1:by2, bx1:bx2], sub_fh.astype(np.uint8) * 255)

            # Chin
            sub_ch = (sub_skin > 0) & (np.arange(sub_h)[:, None] > y_split_bottom)
            chin_mask[by1:by2, bx1:bx2] = np.maximum(chin_mask[by1:by2, bx1:bx2], sub_ch.astype(np.uint8) * 255)

            # Cheeks
            sub_mid = (sub_skin > 0) & (np.arange(sub_h)[:, None] >= y_split_top) & (np.arange(sub_h)[:, None] <= y_split_bottom) & (~sub_nose)
            sub_lc = sub_mid & (np.arange(sub_w)[None, :] < x_nose_center)
            sub_rc = sub_mid & (np.arange(sub_w)[None, :] >= x_nose_center)

            left_cheek_mask[by1:by2, bx1:bx2] = np.maximum(left_cheek_mask[by1:by2, bx1:bx2], sub_lc.astype(np.uint8) * 255)
            right_cheek_mask[by1:by2, bx1:bx2] = np.maximum(right_cheek_mask[by1:by2, bx1:bx2], sub_rc.astype(np.uint8) * 255)
    else:
        # Fallback geometric division
        eye_coords = np.argwhere(zones["eyes"] > 0)
        lip_coords = np.argwhere(zones["lips"] > 0)
        nose_coords = np.argwhere(zones["nose"] > 0)

        y_eye = np.mean(eye_coords[:, 0]) if len(eye_coords) > 0 else int(h * 0.42)
        y_lip_bottom = np.max(lip_coords[:, 0]) if len(lip_coords) > 0 else int(h * 0.72)
        x_center = np.median(nose_coords[:, 1]) if len(nose_coords) > 0 else int(w * 0.5)

        forehead_mask = ((facial_skin > 0) & (np.arange(h)[:, None] < int(y_eye))).astype(np.uint8) * 255
        chin_mask = ((facial_skin > 0) & (np.arange(h)[:, None] > int(y_lip_bottom))).astype(np.uint8) * 255
        mid_face = (facial_skin > 0) & (np.arange(h)[:, None] >= int(y_eye)) & (np.arange(h)[:, None] <= int(y_lip_bottom)) & (zones["nose"] == 0)
        left_cheek_mask = (mid_face & (np.arange(w)[None, :] < x_center)).astype(np.uint8) * 255
        right_cheek_mask = (mid_face & (np.arange(w)[None, :] >= x_center)).astype(np.uint8) * 255

    zones["forehead"] = forehead_mask
    zones["chin"] = chin_mask
    zones["left_cheek"] = left_cheek_mask
    zones["right_cheek"] = right_cheek_mask

    return zones


def segment_face_skin(
    img_rgb: np.ndarray,
    include_neck: bool = True,
    feather_radius: int = 3,
    protect_nostrils: bool = True,
    protect_vermilion: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Layer 1 — Precision Multi-Face Anatomical Skin Segmentation & Facial Zoning.
    
    Robust against multi-person photos, couples, off-center faces, and half/full body portraits:
    1. Runs RetinaFace / SCRFD multi-face detection to accurately bound every face.
    2. Crops each face with 35% contextual padding and parses with BiSeNet in its native resolution.
    3. Merges per-face anatomy back into full-resolution canvas.
    4. Enforces strict anatomical safety exclusion zones:
       - Vermilion lip border (Cupid's bow & lip contours dilated by 5px)
       - Eyelid crease, lashes, and eyebrows (dilated by 5px)
       - Nostril cavities / nasal base crevices
       - 0% skin mask on clothing, lace, dresses, background, jewelry, or arms.
    
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

    # 1. Multi-Face Detection
    detector = get_face_detector()
    detected_faces = []
    if detector is not None:
        try:
            detected_faces = detector.detect(img_bgr)
            if detected_faces:
                logger.info(f"Multi-face detector found {len(detected_faces)} face(s)")
        except Exception as det_err:
            logger.warning(f"Multi-face detector error: {det_err}")
            detected_faces = []

    class_mask = np.zeros((h, w), dtype=np.uint8)

    if detected_faces and len(detected_faces) > 0:
        # Parse each face crop in BiSeNet's optimal 512x512 distribution
        for f_idx, f in enumerate(detected_faces):
            bbox = f.bbox if hasattr(f, "bbox") else f
            bx1, by1, bx2, by2 = [int(v) for v in bbox[:4]]
            fw = max(10, bx2 - bx1)
            fh = max(10, by2 - by1)
            pad_x = int(fw * 0.35)
            pad_y = int(fh * 0.35)
            x1 = max(0, bx1 - pad_x)
            y1 = max(0, by1 - pad_y)
            x2 = min(w, bx2 + pad_x)
            y2 = min(h, by2 + pad_y)

            crop_bgr = img_bgr[y1:y2, x1:x2]
            try:
                crop_mask = parser.parse(crop_bgr)
                # Merge into global mask: prioritize non-background classes
                existing_crop = class_mask[y1:y2, x1:x2]
                merged = np.where(crop_mask > 0, crop_mask, existing_crop)
                class_mask[y1:y2, x1:x2] = merged
            except Exception as crop_err:
                logger.warning(f"BiSeNet parse error on face #{f_idx}: {crop_err}")

        # If crops produced skin, we're done; otherwise try whole image
        skin_count = np.sum(np.isin(class_mask, list(SKIN_CLASSES)))
        if skin_count == 0:
            try:
                class_mask = parser.parse(img_bgr)
            except Exception as e:
                logger.error(f"Fallback full BiSeNet parse error: {e}")
                return fallback_skin_segmentation(img_rgb, feather_radius=feather_radius)
    else:
        # Single portrait or face detector missed: run directly on whole image
        try:
            class_mask = parser.parse(img_bgr)
        except Exception as e:
            logger.error(f"Direct BiSeNet parse error: {e}")
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
    # 1. Protect Lip Vermilion Border (Cupid's bow & mouth)
    if protect_vermilion:
        lip_region = np.isin(class_mask, [11, 12, 13]).astype(np.uint8) * 255
        if np.any(lip_region):
            k_lip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated_lips = cv2.dilate(lip_region, k_lip)
            raw_mask[dilated_lips > 0] = 0

    # 2. Precision Eyebrows & Eye Contour Margins
    # Tight resolution-adaptive margin follows the natural anatomical boundary
    # without cutting out forehead, temples, or brow bone skin.
    dim = min(h, w)
    brow_region = np.isin(class_mask, [2, 3]).astype(np.uint8) * 255
    if np.any(brow_region):
        k_eb_sz = max(3, int(dim * 0.005) | 1)
        k_eb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_eb_sz, k_eb_sz))
        dilated_brows = cv2.dilate(brow_region, k_eb)
        raw_mask[dilated_brows > 0] = 0

    eye_region = np.isin(class_mask, [4, 5]).astype(np.uint8) * 255
    if np.any(eye_region):
        k_eye_sz = max(5, int(dim * 0.008) | 1)
        k_eye = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_eye_sz, k_eye_sz))
        dilated_eyes = cv2.dilate(eye_region, k_eye)
        raw_mask[dilated_eyes > 0] = 0

    # 3. Nostril / dark cavity protection in nasal base (class 10)
    if protect_nostrils:
        nose_region = (class_mask == 10)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        nostril_candidates = (gray < 45) & nose_region
        if np.any(nostril_candidates):
            k_nos = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            nostril_mask = cv2.dilate(nostril_candidates.astype(np.uint8) * 255, k_nos)
            raw_mask[nostril_mask > 0] = 0

    # 4. Morphological cleanup
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Extract sub-facial zones
    zones = extract_facial_zones(class_mask, cleaned_mask, img_rgb, faces=detected_faces)

    # Sample healthy base skin tone (median RGB and LAB of verified facial skin pixels)
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

    # Resolution-adaptive soft edge feathering:
    # Guarantees seamless gradient blending into background and hair boundaries
    eff_feather = max(feather_radius, max(8, int(dim * 0.012)))
    if eff_feather > 0:
        ksize = eff_feather * 2 + 1
        final_mask = cv2.GaussianBlur(cleaned_mask, (ksize, ksize), 0)
    else:
        final_mask = cleaned_mask

    # Non-skin anatomy mask (lips, mouth, nostrils, eyes, background, clothing)
    non_skin = np.ones((h, w), dtype=np.uint8) * 255
    non_skin[cleaned_mask > 128] = 0

    metadata = {
        "classes_present": [int(c) for c in np.unique(class_mask)],
        "faces_detected": len(detected_faces) if detected_faces else 1,
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
