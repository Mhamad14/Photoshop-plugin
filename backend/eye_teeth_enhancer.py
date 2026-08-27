import logging
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any

logger = logging.getLogger("eye_teeth_enhancer")


def enhance_eyes_and_teeth(
    img_rgb: np.ndarray,
    teeth_whiten_strength: float = 0.5,
    eye_brighten_strength: float = 0.5,
    iris_sparkle_strength: float = 0.35,
    feather_radius: int = 3
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Anatomically Realistic AI Eye & Teeth Retouching Engine.
    
    1. Natural Enamel Teeth Whitening:
       - Preserves interdental shadows (spaces between teeth) to prevent "flat chiclet" look.
       - Parabolic vertical shading keeps natural gumline depth and incisal edge translucency.
       - Selectively neutralizes yellow/brown tea & coffee stains (CIE LAB b* suppression).
       
    2. Realistic Eye Sclera & Iris Enhancer:
       - Preserves natural shadow falloff at eye corners (canthus) to prevent "radioactive / alien" eyes.
       - Selectively removes red capillaries & bloodshot veins (a* suppression).
       - Sharpens iris micro-contrast and enhances specular catchlights without washing out the pupil.
       
    Returns:
        composited_rgb: uint8 HxWx3
        patch_alpha: uint8 HxW mask for non-destructive layer placement
        stats: dict of enhancements applied
    """
    from face_segmenter import get_bisenet_parser
    parser = get_bisenet_parser()
    h, w, _ = img_rgb.shape

    if parser is None:
        return img_rgb.copy(), np.zeros((h, w), dtype=np.uint8), {"status": "parser_unavailable"}

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    try:
        class_mask = parser.parse(img_bgr)
    except Exception as e:
        logger.error(f"Eye/Teeth parse error: {e}")
        return img_rgb.copy(), np.zeros((h, w), dtype=np.uint8), {"status": "parse_failed"}

    # Class 11 = mouth/teeth, Class 4 = left eye, Class 5 = right eye
    mouth_mask = (class_mask == 11).astype(np.uint8) * 255
    eyes_mask = (np.isin(class_mask, [4, 5])).astype(np.uint8) * 255

    enhanced_rgb = img_rgb.copy().astype(np.float32)
    output_alpha = np.zeros((h, w), dtype=np.uint8)

    # -------------------------------------------------------------
    # 1. ANATOMICALLY REALISTIC TEETH WHITENING
    # -------------------------------------------------------------
    teeth_applied = False
    if teeth_whiten_strength > 0 and np.sum(mouth_mask > 0) > 40:
        mouth_rgb = img_rgb.copy()
        mouth_hsv = cv2.cvtColor(mouth_rgb, cv2.COLOR_RGB2HSV)
        mouth_lab = cv2.cvtColor(mouth_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        v_chan = mouth_hsv[:, :, 2]
        s_chan = mouth_hsv[:, :, 1]
        l_chan, a_chan, b_chan = cv2.split(mouth_lab)

        # Candidate teeth pixels: high luminance inside mouth, not red/purple gums/tongue
        teeth_candidate = (mouth_mask > 0) & (v_chan > 105) & (s_chan < 165) & (a_chan < 160)

        if np.sum(teeth_candidate) > 20:
            teeth_applied = True
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            teeth_clean = cv2.morphologyEx(teeth_candidate.astype(np.uint8) * 255, cv2.MORPH_OPEN, k)

            # Interdental shadow protection (detect gaps/crevices between teeth via morphological gradient)
            grad = cv2.morphologyEx(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), cv2.MORPH_GRADIENT, k)
            crevice_protect = np.clip(1.0 - (grad.astype(np.float32) / 70.0), 0.25, 1.0)

            # Parabolic vertical enamel depth falloff
            # Teeth naturally have a gradient: darker near gums, bright in center, translucent at tips
            teeth_coords = np.argwhere(teeth_clean > 0)
            if len(teeth_coords) > 0:
                y_min, y_max = np.min(teeth_coords[:, 0]), np.max(teeth_coords[:, 0])
                y_range = max(1, y_max - y_min)
                y_indices = np.arange(h)[:, None]
                # Normalized vertical position within teeth region (0 to 1)
                y_norm = np.clip((y_indices - y_min) / y_range, 0.0, 1.0)
                # Parabolic bell curve: peaks at center (0.5), gently softens at gums (0.0) and edges (1.0)
                vertical_falloff = 4.0 * y_norm * (1.0 - y_norm)
                vertical_falloff = np.clip(vertical_falloff * 0.7 + 0.3, 0.3, 1.0)
            else:
                vertical_falloff = np.ones((h, w), dtype=np.float32)

            teeth_soft = cv2.GaussianBlur(teeth_clean, (5, 5), 0).astype(np.float32) / 255.0
            teeth_weight = teeth_soft * crevice_protect * vertical_falloff

            whiten_factor = min(1.0, teeth_whiten_strength)

            # Desaturate yellow tea/coffee cast (pull b* towards neutral 128)
            b_corrected = b_chan * (1.0 - whiten_factor * 0.75 * teeth_weight) + 128.0 * (whiten_factor * 0.75 * teeth_weight)
            a_corrected = a_chan * (1.0 - whiten_factor * 0.40 * teeth_weight) + 128.0 * (whiten_factor * 0.40 * teeth_weight)

            # Natural luminance boost (avoiding chalky flat look)
            l_boost = l_chan + (255.0 - l_chan) * (whiten_factor * 0.22) * teeth_weight
            l_corrected = l_boost.clip(0, 255)

            teeth_merged = cv2.merge([l_corrected, a_corrected, b_corrected]).astype(np.uint8)
            teeth_rgb = cv2.cvtColor(teeth_merged, cv2.COLOR_LAB2RGB).astype(np.float32)

            t_factor = (teeth_weight * whiten_factor)[:, :, None]
            enhanced_rgb = enhanced_rgb * (1.0 - t_factor) + teeth_rgb * t_factor
            output_alpha = np.maximum(output_alpha, (teeth_weight * 255).astype(np.uint8))

    # -------------------------------------------------------------
    # 2. REALISTIC EYE SCLERA & IRIS ENHANCER
    # -------------------------------------------------------------
    eyes_applied = False
    if (eye_brighten_strength > 0 or iris_sparkle_strength > 0) and np.sum(eyes_mask > 0) > 40:
        eyes_applied = True
        eye_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_eye, a_eye, b_eye = cv2.split(eye_lab)

        # Sclera (whites) vs Iris separation
        sclera_candidate = (eyes_mask > 0) & (l_eye > 130)
        iris_candidate = (eyes_mask > 0) & ~sclera_candidate

        # 2a. Sclera Whitening (neutralize red veins without bleaching corner shadows)
        if np.sum(sclera_candidate) > 15 and eye_brighten_strength > 0:
            s_soft = cv2.GaussianBlur(sclera_candidate.astype(np.uint8) * 255, (3, 3), 0).astype(np.float32) / 255.0
            
            # Corner falloff: distance transform preserves natural shadows in eye corners
            dist_map = cv2.distanceTransform(sclera_candidate.astype(np.uint8), cv2.DIST_L2, 3)
            max_dist = np.max(dist_map) if np.max(dist_map) > 0 else 1.0
            corner_falloff = np.clip(dist_map / (max_dist * 0.6 + 1e-5), 0.25, 1.0)

            s_weight = s_soft * corner_falloff
            s_strength = min(1.0, eye_brighten_strength)

            # Suppress bloodshot capillaries (a* channel)
            a_sclera = a_eye * (1.0 - s_strength * 0.80 * s_weight) + 128.0 * (s_strength * 0.80 * s_weight)
            b_sclera = b_eye * (1.0 - s_strength * 0.55 * s_weight) + 128.0 * (s_strength * 0.55 * s_weight)
            
            # Gentle luminance lift (never blow out sclera)
            l_sclera = (l_eye + (255.0 - l_eye) * (s_strength * 0.18) * s_weight).clip(0, 255)

            sclera_lab = cv2.merge([l_sclera, a_sclera, b_sclera]).astype(np.uint8)
            sclera_rgb = cv2.cvtColor(sclera_lab, cv2.COLOR_LAB2RGB).astype(np.float32)

            s_factor = (s_weight * s_strength)[:, :, None]
            enhanced_rgb = enhanced_rgb * (1.0 - s_factor) + sclera_rgb * s_factor
            output_alpha = np.maximum(output_alpha, (s_weight * 255).astype(np.uint8))

        # 2b. Iris Sparkle & Catchlight Enhancement (Micro-contrast without pupil wash)
        if np.sum(iris_candidate) > 15 and iris_sparkle_strength > 0:
            iris_soft = cv2.GaussianBlur(iris_candidate.astype(np.uint8) * 255, (3, 3), 0).astype(np.float32) / 255.0
            i_strength = min(1.0, iris_sparkle_strength)

            # Protect dark center pupil from over-brightening
            pupil_protect = np.clip((l_eye - 25.0) / 40.0, 0.0, 1.0)

            # High-pass micro-contrast for iris radial fibers & catchlights
            blurred_iris = cv2.GaussianBlur(enhanced_rgb, (0, 0), 2.2)
            high_pass = enhanced_rgb - blurred_iris

            sparkled = enhanced_rgb + high_pass * (i_strength * 1.5) * (pupil_protect[:, :, None])
            sparkled = sparkled.clip(0, 255)

            i_factor = (iris_soft * i_strength * pupil_protect)[:, :, None]
            enhanced_rgb = enhanced_rgb * (1.0 - i_factor) + sparkled * i_factor
            output_alpha = np.maximum(output_alpha, (iris_soft * 255).astype(np.uint8))

    # Apply soft boundary feathering to output alpha mask
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
    teeth_whiten_strength: float = 0.5,
    eye_brighten_strength: float = 0.5,
    iris_sparkle_strength: float = 0.35,
    feather_radius: int = 3
) -> Image.Image:
    """Returns transparent RGBA patch with natural whitened teeth and brightened eyes."""
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
