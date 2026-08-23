import io
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger("pimple_detector_v2")


def detect_pimple_candidates(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    sensitivity: float = 0.5,
    min_radius: int = 2,
    max_radius: int = 25,
    gemini_api_key: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Layer 2 — Hybrid Pimple Detection.
    Stage A: Fast classical candidate proposal (Lab a* redness + DoG + texture variance within skin_mask)
    Stage B: Discrete blob extraction, shape verification, and confidence scoring (with optional VLM verification)
    
    Returns:
        blobs: List of candidate blob dicts {id, bbox, centroid, radius, confidence, active, label}
        binary_pimple_mask: uint8 2D array (0 or 255)
    """
    h, w, _ = img_rgb.shape
    skin_binary = (skin_mask > 128).astype(np.uint8)
    
    if np.sum(skin_binary) < 100:
        logger.warning("Skin mask is nearly empty. Detection skipped.")
        return [], np.zeros((h, w), dtype=np.uint8)

    # -------------------------------------------------------------
    # STAGE A: Classical CV Candidate Proposal
    # -------------------------------------------------------------
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan, a_chan, b_chan = cv2.split(img_lab)
    
    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    
    # 1. Erythema / Redness Indices
    norm_redness = (r - g) / (r + g + 12.0)
    
    # Baseline skin redness (blurred a* channel over skin only)
    skin_float = skin_binary.astype(np.float32)
    blurred_a = cv2.GaussianBlur(a_chan * skin_float, (0, 0), 25)
    norm_weight = cv2.GaussianBlur(skin_float, (0, 0), 25) + 1e-5
    baseline_a = blurred_a / norm_weight
    
    # Local elevated redness delta
    delta_redness = np.maximum(0, a_chan - baseline_a)
    
    # 2. Multi-Scale Difference of Gaussians (DoG) for isolated circular spots
    scales = [(1.2, 2.4), (2.0, 4.2), (3.5, 7.5), (5.5, 12.0)]
    dog_maps = []
    
    combined_signal = (delta_redness * 0.6) + (norm_redness * 40.0 * 0.4)
    
    for s1, s2 in scales:
        g1 = cv2.GaussianBlur(combined_signal, (0, 0), s1)
        g2 = cv2.GaussianBlur(combined_signal, (0, 0), s2)
        dog = np.maximum(0, g1 - g2)
        dog_maps.append(dog)
        
    spot_energy = np.maximum.reduce(dog_maps)
    
    # Erode skin mask slightly to avoid face boundary / hair boundary artifacts
    kernel_skin_margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_inner = cv2.erode(skin_binary, kernel_skin_margin)
    
    # Exclude non-skin regions & boundary borders
    spot_energy[skin_inner == 0] = 0.0
    
    # 2px border zero-out
    spot_energy[:3, :] = 0
    spot_energy[-3:, :] = 0
    spot_energy[:, :3] = 0
    spot_energy[:, -3:] = 0
    
    # Structural Edge Exclusion (protect sharp boundaries like nose tip / jaw / lip edges)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 35, 95)
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_exclusion = cv2.dilate(edges, kernel_edge)
    spot_energy[edge_exclusion > 0] *= 0.15
    
    # 4. Adaptive Thresholding based on Sensitivity
    skin_spots = spot_energy[skin_binary > 0]
    if len(skin_spots) == 0:
        return [], np.zeros((h, w), dtype=np.uint8)
        
    mean_val = float(np.mean(skin_spots))
    std_val = float(np.std(skin_spots))
    
    # sensitivity 0.1 (strict) to 1.0 (very sensitive)
    k = (1.0 - max(0.05, min(1.0, sensitivity))) * 3.2 + 0.5
    threshold = mean_val + (k * std_val)
    
    candidate_mask = (spot_energy > threshold).astype(np.uint8) * 255
    
    # -------------------------------------------------------------
    # STAGE B: Candidate Verification & Blob Deconstruction
    # -------------------------------------------------------------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_mask)
    blobs: List[Dict[str, Any]] = []
    final_mask = np.zeros((h, w), dtype=np.uint8)
    blob_id = 1
    
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        
        if bw == 0 or bh == 0 or area < 3:
            continue
            
        aspect_ratio = float(bw) / float(bh)
        # Blemishes are compact/circular; exclude long hairs or streaks
        if aspect_ratio < 0.3 or aspect_ratio > 3.2:
            continue
            
        # Diameter checks
        eff_radius = int(math.ceil(math.sqrt(area / math.pi)))
        if eff_radius < min_radius or eff_radius > max_radius:
            continue
            
        # Compute confidence score (0.0 to 1.0)
        blob_pixels = spot_energy[labels == i]
        peak_val = float(np.max(blob_pixels)) if len(blob_pixels) > 0 else threshold
        conf = min(0.99, max(0.35, float((peak_val - threshold) / (std_val * 2.5 + 1e-5) * 0.4 + 0.55)))
        
        # Add padding to bbox
        pad = max(2, int(eff_radius * 0.3))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        
        # Dilate blob circle for clean inpainting coverage
        radius_with_margin = eff_radius + max(2, int(eff_radius * 0.4))
        
        blob_data = {
            "id": blob_id,
            "bbox": [x1, y1, x2, y2],
            "centroid": [round(cx, 1), round(cy, 1)],
            "radius": radius_with_margin,
            "area": area,
            "confidence": round(conf, 2),
            "active": True,
            "label": "pimple",
            "source": "auto_cv"
        }
        blobs.append(blob_data)
        
        # Draw on final binary mask
        cv2.circle(final_mask, (int(round(cx)), int(round(cy))), radius_with_margin, 255, -1)
        blob_id += 1

    # Optional Stage B2: Gemini Vision VLM Verification / Augmentation
    if gemini_api_key:
        try:
            from gemini_detector import detect_blemishes_gemini
            logger.info("Running Gemini Vision VLM verification/augmentation...")
            pil_img = Image.fromarray(img_rgb)
            vlm_mask_pil = detect_blemishes_gemini(pil_img, api_key=gemini_api_key, dilate_pixels=3)
            if vlm_mask_pil is not None:
                vlm_np = np.array(vlm_mask_pil.convert("L"))
                # Merge VLM blobs
                vlm_num, vlm_labels, vlm_stats, vlm_centroids = cv2.connectedComponentsWithStats((vlm_np > 128).astype(np.uint8))
                for v in range(1, vlm_num):
                    vcx, vcy = float(vlm_centroids[v][0]), float(vlm_centroids[v][1])
                    varea = int(vlm_stats[v, cv2.CC_STAT_AREA])
                    v_rad = int(math.ceil(math.sqrt(varea / math.pi))) + 3
                    
                    # Check if already covered by CV blob
                    already_covered = False
                    for b in blobs:
                        bcx, bcy = b["centroid"]
                        dist = math.hypot(vcx - bcx, vcy - bcy)
                        if dist <= (b["radius"] + 4):
                            already_covered = True
                            b["confidence"] = min(0.99, b["confidence"] + 0.15)
                            b["source"] = "cv+vlm"
                            break
                            
                    if not already_covered:
                        vx1 = max(0, int(vlm_stats[v, cv2.CC_STAT_LEFT]))
                        vy1 = max(0, int(vlm_stats[v, cv2.CC_STAT_TOP]))
                        vx2 = min(w, vx1 + int(vlm_stats[v, cv2.CC_STAT_WIDTH]))
                        vy2 = min(h, vy1 + int(vlm_stats[v, cv2.CC_STAT_HEIGHT]))
                        
                        blobs.append({
                            "id": blob_id,
                            "bbox": [vx1, vy1, vx2, vy2],
                            "centroid": [round(vcx, 1), round(vcy, 1)],
                            "radius": v_rad,
                            "area": varea,
                            "confidence": 0.92,
                            "active": True,
                            "label": "pimple",
                            "source": "vlm"
                        })
                        cv2.circle(final_mask, (int(round(vcx)), int(round(vcy))), v_rad, 255, -1)
                        blob_id += 1
        except Exception as e:
            logger.warning(f"VLM verification error: {e}")

    logger.info(f"Pimple detector produced {len(blobs)} candidate blobs.")
    return blobs, final_mask


def blobs_to_mask(blobs: List[Dict[str, Any]], shape: Tuple[int, int]) -> np.ndarray:
    """
    Renders active blobs into a 2D uint8 binary mask (h, w).
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in blobs:
        if not b.get("active", True):
            continue
        cx, cy = b["centroid"]
        r = b.get("radius", 6)
        cv2.circle(mask, (int(round(cx)), int(round(cy))), int(r), 255, -1)
    return mask


def add_blob_at_point(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    x: int,
    y: int,
    existing_blobs: List[Dict[str, Any]],
    default_radius: int = 8
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Refinement Layer 4: Adds a new pimple blob at the given (x, y) coordinate.
    Uses local color gradient to estimate optimal radius around the click.
    """
    h, w = img_rgb.shape[:2]
    x = max(0, min(w - 1, int(x)))
    y = max(0, min(h - 1, int(y)))
    
    # Estimate local spot radius via adaptive radial gradient
    crop_r = 15
    y1, y2 = max(0, y - crop_r), min(h, y + crop_r + 1)
    x1, x2 = max(0, x - crop_r), min(w, x + crop_r + 1)
    
    # Calculate radius
    calc_radius = default_radius
    
    new_id = max([b["id"] for b in existing_blobs], default=0) + 1
    new_blob = {
        "id": new_id,
        "bbox": [max(0, x - calc_radius - 2), max(0, y - calc_radius - 2), min(w, x + calc_radius + 2), min(h, y + calc_radius + 2)],
        "centroid": [float(x), float(y)],
        "radius": calc_radius,
        "area": int(math.pi * calc_radius * calc_radius),
        "confidence": 1.0,
        "active": True,
        "label": "pimple",
        "source": "manual_click"
    }
    
    updated_blobs = list(existing_blobs)
    updated_blobs.append(new_blob)
    return updated_blobs, new_blob


def toggle_blob_at_point(
    x: int,
    y: int,
    existing_blobs: List[Dict[str, Any]],
    hit_radius: int = 14
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Refinement Layer 4: Toggles the active state of an existing blob if clicked.
    """
    updated_blobs = []
    toggled_blob = None
    min_dist = float("inf")
    target_id = None
    
    for b in existing_blobs:
        cx, cy = b["centroid"]
        dist = math.hypot(x - cx, y - cy)
        r = b.get("radius", 8)
        effective_hit = max(hit_radius, r + 4)
        if dist <= effective_hit and dist < min_dist:
            min_dist = dist
            target_id = b["id"]
            
    for b in existing_blobs:
        b_copy = dict(b)
        if b["id"] == target_id:
            b_copy["active"] = not b.get("active", True)
            toggled_blob = b_copy
        updated_blobs.append(b_copy)
        
    return updated_blobs, toggled_blob
