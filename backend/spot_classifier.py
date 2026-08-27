import logging
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("spot_classifier")


def classify_spot(
    img_rgb: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    base_skin_lab: np.ndarray
) -> Dict[str, Any]:
    """
    Fitzpatrick-Adaptive Dermatological Spot Classifier:
    Differentiates between temporary inflammatory acne (pimples, pustules) vs permanent melanin marks (moles, beauty marks) vs freckles.
    
    Principles:
    1. Erythema Index (EI): Measures delta red inflammation (a* excess) relative to individual base tone.
    2. Melanin Depression Index: Scales dynamically relative to base skin L* (adapts for Fitzpatrick Types I - VI).
    3. Morphological Circularity: Moles have regular, well-defined circular margins and even internal pigment;
       acne lesions have diffuse erythema halos and irregular inflamed borders.
    """
    h, w, _ = img_rgb.shape
    r = max(3, radius)
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w, cx + r + 1)
    y2 = min(h, cy + r + 1)

    crop = img_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return {"type": "pimple", "confidence": 0.5, "is_mole": False, "is_freckle": False}

    crop_lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_c, a_c, b_c = cv2.split(crop_lab)

    base_l = float(base_skin_lab[0])
    base_a = float(base_skin_lab[1])

    center_l = float(np.mean(l_c))
    center_a = float(np.mean(a_c))

    erythema_elevation = center_a - base_a  # Inflamed acne excess
    lum_drop = base_l - center_l            # Melanin darkness depression

    # Fitzpatrick tone adaptive thresholding
    # On fair skin (L* > 180), melanin drop > 25 is a mole; on darker skin (L* < 130), melanin drop > 16 is a mole
    mole_lum_threshold = max(14.0, base_l * 0.18)
    erythema_threshold = 6.5 if base_l > 160 else 4.5

    # Circularity & border regularity analysis
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circularity = 0.5
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter > 0:
            circularity = float(4 * np.pi * (area / (perimeter * perimeter)))

    # Classification logic:
    # 1. Acne / Pimple: Significant red erythema elevation (delta a* > threshold)
    # 2. Mole / Beauty mark: High melanin depression (delta L* > threshold) WITHOUT erythema excess (delta a* <= 4.0)
    # 3. Freckle: Small, light melanin depression without erythema excess
    if erythema_elevation > erythema_threshold:
        # Inflamed red acne lesion / pimple / pustule
        spot_type = "pimple"
        is_mole = False
        is_freckle = False
    elif lum_drop > mole_lum_threshold and circularity > 0.35 and erythema_elevation <= 4.5:
        # Dark, circular, stable melanin pigment without erythema -> Permanent Mole / Beauty Mark
        spot_type = "mole"
        is_mole = True
        is_freckle = False
    elif lum_drop > (mole_lum_threshold * 0.40) and r <= 6 and erythema_elevation <= 3.5:
        # Small light brown ephelis -> Freckle
        spot_type = "freckle"
        is_mole = False
        is_freckle = True
    else:
        spot_type = "pimple"
        is_mole = False
        is_freckle = False

    return {
        "type": spot_type,
        "is_mole": is_mole,
        "is_freckle": is_freckle,
        "erythema": float(round(erythema_elevation, 1)),
        "lum_drop": float(round(lum_drop, 1)),
        "circularity": float(round(circularity, 2))
    }


def filter_blobs_by_preference(
    blobs: List[Dict[str, Any]],
    img_rgb: np.ndarray,
    preserve_moles: bool = True,
    preserve_freckles: bool = False
) -> List[Dict[str, Any]]:
    """
    Filters detected blemish blobs according to user preferences (preserving beauty marks/freckles).
    """
    if not preserve_moles and not preserve_freckles:
        return blobs

    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    base_skin_lab = np.median(img_lab.reshape(-1, 3), axis=0)

    filtered = []
    for blob in blobs:
        cx = int(blob.get("x", blob.get("cx", blob.get("centroid", [0, 0])[0])))
        cy = int(blob.get("y", blob.get("cy", blob.get("centroid", [0, 0])[1])))
        r = int(blob.get("radius", blob.get("r", 6)))

        classification = classify_spot(img_rgb, cx, cy, r, base_skin_lab)

        if preserve_moles and classification["is_mole"]:
            continue
        if preserve_freckles and classification["is_freckle"]:
            continue

        blob_copy = dict(blob)
        blob_copy["spot_type"] = classification["type"]
        filtered.append(blob_copy)

    return filtered
