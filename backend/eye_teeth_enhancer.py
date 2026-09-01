import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("eye_teeth_enhancer")


def segment_teeth_enamel_robust(
    img_rgb: np.ndarray,
    class_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full Dental Arch & Enamel Colorimetric Segmenter.
    Accurately captures Central Incisors, Lateral Incisors, Canines, Premolars,
    and shaded Back/Side Molars across full portraits and macro smile shots.
    
    Returns:
        teeth_mask: uint8 2D mask (0 to 255) containing the complete dental arch
        gums_lips_mask: uint8 2D mask of gingiva (gums) and lips
    """
    h, w, _ = img_rgb.shape
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan, a_chan, b_chan = cv2.split(img_lab)

    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h_chan, s_chan, v_chan = cv2.split(img_hsv)

    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)

    red_ratio = (r - g) / (r + g + 18.0)

    # 1. Gums & Lip Tissue Detection (Gingiva and Mucosa)
    # Gums and lips have high red chroma (a* >= 148, red_ratio > 0.17)
    is_gum_or_lip = (a_chan >= 148.0) | (red_ratio > 0.17) | (h_chan > 168) | (h_chan < 12)
    if class_mask is not None:
        is_gum_or_lip = is_gum_or_lip | np.isin(class_mask, [12, 13])

    k_gum = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_gums_lips = cv2.dilate(is_gum_or_lip.astype(np.uint8) * 255, k_gum)

    # 2. Deep Throat / Oral Cavity Blackness (Strictly pitch black areas, not side teeth)
    dark_cavity = (l_chan < 38.0) | (v_chan < 35)

    # 3. Complete Dental Arch Candidate Signature:
    # - Enamel has luminance L* >= 46 (captures shadowed back molars & side premolars)
    # - Low red chroma (a* < 147, not red gums or pink lips)
    # - Low to moderate saturation (S < 175)
    # - Not pitch dark oral cavity and not gums/lips
    enamel_candidates = (
        (l_chan >= 46.0) &
        (v_chan >= 44) &
        (a_chan < 147.0) &
        (s_chan < 175) &
        (~dark_cavity) &
        (dilated_gums_lips == 0)
    )

    # If BiSeNet mouth mask (class 11) is available, intersect with generous mouth dilation
    if class_mask is not None:
        mouth_region = (class_mask == 11)
        if np.sum(mouth_region) > 80:
            k_mouth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            expanded_mouth = cv2.dilate(mouth_region.astype(np.uint8) * 255, k_mouth) > 0
            enamel_candidates = enamel_candidates & expanded_mouth

    # 4. Morphological Cleaning & Arch Connectivity
    enamel_uint8 = enamel_candidates.astype(np.uint8) * 255
    k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_enamel = cv2.morphologyEx(enamel_uint8, cv2.MORPH_CLOSE, k_clean)

    # Keep all valid dental components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_enamel)
    final_enamel = np.zeros((h, w), dtype=np.uint8)

    min_tooth_area = max(10, int(h * w * 0.00008))
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_tooth_area:
            final_enamel[labels == i] = 255

    return final_enamel, dilated_gums_lips


def enhance_eyes_and_teeth(
    img_rgb: np.ndarray,
    teeth_whiten_strength: float = 0.65,
    eye_brighten_strength: float = 0.45,
    iris_sparkle_strength: float = 0.35,
    feather_radius: int = 3
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Full Dental Arch AI Teeth Whitening & Eye Enhancement Engine.
    Cleans and whitens front incisors, lateral canines, and back/side molars.
    """
    h, w, _ = img_rgb.shape
    from face_segmenter import get_bisenet_parser
    parser = get_bisenet_parser()

    class_mask = None
    if parser is not None:
        try:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            class_mask = parser.parse(img_bgr)
        except Exception as e:
            logger.warning("BiSeNet parse for eye/teeth: %s", e)

    enhanced_rgb = img_rgb.copy().astype(np.float32)
    output_alpha = np.zeros((h, w), dtype=np.uint8)

    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan, a_chan, b_chan = cv2.split(img_lab)

    teeth_applied = False

    # -------------------------------------------------------------
    # 1. FULL DENTAL ARCH TEETH WHITENING (Front, Sides & Back Molars)
    # -------------------------------------------------------------
    if teeth_whiten_strength > 0:
        teeth_mask, gums_mask = segment_teeth_enamel_robust(img_rgb, class_mask=class_mask)
        enamel_pixel_count = np.sum(teeth_mask > 0)

        if enamel_pixel_count > 20:
            teeth_applied = True
            
            # Interdental crevice protection
            k_crev = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k_crev)
            crevice_protect = np.clip(1.0 - (grad.astype(np.float32) / 60.0), 0.35, 1.0)

            # Soft edge feathering
            teeth_soft = cv2.GaussianBlur(teeth_mask, (5, 5), 0).astype(np.float32) / 255.0
            teeth_weight = teeth_soft * crevice_protect

            w_strength = max(0.0, min(1.0, teeth_whiten_strength))

            # 1a. Neutralize Yellow & Brown Chroma (Front, sides & back)
            target_b = 128.0 + (b_chan - 128.0) * (1.0 - w_strength * 0.94)
            b_corrected = b_chan * (1.0 - teeth_weight * w_strength * 0.94) + target_b * (teeth_weight * w_strength * 0.94)

            # 1b. Neutralize Orange/Tartar Red Tint (a* channel)
            target_a = 128.0 + (a_chan - 128.0) * (1.0 - w_strength * 0.60)
            a_corrected = a_chan * (1.0 - teeth_weight * w_strength * 0.60) + target_a * (teeth_weight * w_strength * 0.60)

            # 1c. Luminance Lift across Front & Side/Back Teeth
            l_lift = (255.0 - l_chan) * (w_strength * 0.34) * teeth_weight
            l_corrected = (l_chan + l_lift).clip(0, 255)

            # 1d. Reconstruct Whitened RGB
            teeth_merged = cv2.merge([l_corrected, a_corrected, b_corrected]).clip(0, 255).astype(np.uint8)
            teeth_rgb = cv2.cvtColor(teeth_merged, cv2.COLOR_LAB2RGB).astype(np.float32)

            t_factor = (teeth_weight * w_strength)[:, :, None]
            enhanced_rgb = enhanced_rgb * (1.0 - t_factor) + teeth_rgb * t_factor
            output_alpha = np.maximum(output_alpha, (teeth_weight * 255).astype(np.uint8))

    # -------------------------------------------------------------
    # 2. REALISTIC EYE SCLERA & IRIS ENHANCER
    # -------------------------------------------------------------
    eyes_applied = False
    if class_mask is not None and (eye_brighten_strength > 0 or iris_sparkle_strength > 0):
        eyes_mask = (np.isin(class_mask, [4, 5])).astype(np.uint8) * 255
        if np.sum(eyes_mask > 0) > 40:
            eyes_applied = True
            eye_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            l_eye, a_eye, b_eye = cv2.split(eye_lab)

            eye_l_vals = l_eye[eyes_mask > 0]
            if len(eye_l_vals) > 10:
                sclera_thresh = max(95.0, min(145.0, float(np.percentile(eye_l_vals, 55))))
            else:
                sclera_thresh = 125.0

            sclera_candidate = (eyes_mask > 0) & (l_eye >= sclera_thresh)
            iris_candidate = (eyes_mask > 0) & (l_eye < sclera_thresh)

            if np.sum(sclera_candidate) > 10 and eye_brighten_strength > 0:
                s_soft = cv2.GaussianBlur(sclera_candidate.astype(np.uint8) * 255, (3, 3), 0).astype(np.float32) / 255.0
                dist_map = cv2.distanceTransform(sclera_candidate.astype(np.uint8), cv2.DIST_L2, 3)
                max_dist = np.max(dist_map) if np.max(dist_map) > 0 else 1.0
                corner_falloff = np.clip(dist_map / (max_dist * 0.6 + 1e-5), 0.30, 1.0)

                s_weight = s_soft * corner_falloff
                s_strength = min(1.0, eye_brighten_strength)

                a_sclera = a_eye * (1.0 - s_strength * 0.80 * s_weight) + 128.0 * (s_strength * 0.80 * s_weight)
                b_sclera = b_eye * (1.0 - s_strength * 0.55 * s_weight) + 128.0 * (s_strength * 0.55 * s_weight)
                l_sclera = (l_eye + (255.0 - l_eye) * (s_strength * 0.18) * s_weight).clip(0, 255)

                sclera_lab = cv2.merge([l_sclera, a_sclera, b_sclera]).clip(0, 255).astype(np.uint8)
                sclera_rgb = cv2.cvtColor(sclera_lab, cv2.COLOR_LAB2RGB).astype(np.float32)

                s_factor = (s_weight * s_strength)[:, :, None]
                enhanced_rgb = enhanced_rgb * (1.0 - s_factor) + sclera_rgb * s_factor
                output_alpha = np.maximum(output_alpha, (s_weight * 255).astype(np.uint8))

            if np.sum(iris_candidate) > 15 and iris_sparkle_strength > 0:
                iris_soft = cv2.GaussianBlur(iris_candidate.astype(np.uint8) * 255, (3, 3), 0).astype(np.float32) / 255.0
                i_strength = min(1.0, iris_sparkle_strength)
                pupil_protect = np.clip((l_eye - 25.0) / 40.0, 0.0, 1.0)

                blurred_iris = cv2.GaussianBlur(enhanced_rgb, (0, 0), 2.2)
                high_pass = enhanced_rgb - blurred_iris
                sparkled = (enhanced_rgb + high_pass * (i_strength * 1.5) * (pupil_protect[:, :, None])).clip(0, 255)

                i_factor = (iris_soft * i_strength * pupil_protect)[:, :, None]
                enhanced_rgb = enhanced_rgb * (1.0 - i_factor) + sparkled * i_factor
                output_alpha = np.maximum(output_alpha, (iris_soft * 255).astype(np.uint8))

    if feather_radius > 0:
        ksize = feather_radius * 2 + 1
        output_alpha = cv2.GaussianBlur(output_alpha, (ksize, ksize), 0)

    final_rgb = enhanced_rgb.clip(0, 255).astype(np.uint8)

    return final_rgb, output_alpha, {
        "teeth_whitened": teeth_applied,
        "eyes_brightened": eyes_applied
    }


def create_eye_teeth_rgba_patch(
    img_rgb: np.ndarray,
    teeth_whiten_strength: float = 0.65,
    eye_brighten_strength: float = 0.45,
    iris_sparkle_strength: float = 0.35,
    feather_radius: int = 3
) -> Image.Image:
    composited_rgb, alpha, _ = enhance_eyes_and_teeth(
        img_rgb,
        teeth_whiten_strength=teeth_whiten_strength,
        eye_brighten_strength=eye_brighten_strength,
        iris_sparkle_strength=iris_sparkle_strength,
        feather_radius=feather_radius
    )
    r = Image.fromarray(composited_rgb[:, :, 0])
    g = Image.fromarray(composited_rgb[:, :, 1])
    b = Image.fromarray(composited_rgb[:, :, 2])
    a = Image.fromarray(alpha)
    return Image.merge("RGBA", (r, g, b, a))
