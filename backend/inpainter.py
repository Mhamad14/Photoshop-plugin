"""
Shared neural inpainting engine for the AI Retouching plugin.

All heal paths (preview, apply-heal, spot healer) import LaMa tiled inpainting
from here so they stay in sync. Previously the spot healer imported a missing
module and silently fell back to Telea diffusion inpainting, which produced
flat, smudgy patches instead of neural skin reconstruction.
"""

import logging

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("inpainter")


def inpaint_with_context_tiling(
    model,
    img_rgb: np.ndarray,
    mask_gray: np.ndarray,
    max_tile_size: int = 768,
    context_pad: int = 80
) -> np.ndarray:
    """
    Context-Aware Tiled Neural Inpainting with Adaptive Context Padding.

    - Small images run one direct LaMa pass (the quality reference used by the
      live preview).
    - Large images cluster NEARBY spots only (small merge distance). A large
      merge kernel would fuse every spot on the face into one giant cluster and
      force LaMa to inpaint a full-resolution face crop far outside its ~512px
      training distribution, which produces mushy, low-contrast fill.
    - Cluster crops run at native resolution while they fit max_tile_size;
      oversized clusters are downscaled into distribution and their masked
      pixels are upscaled back with Lanczos.
    - Context padding scales adaptively: small spots get tighter context (faster,
      more relevant), large clusters get wider context (better structural coherence).
    """
    h, w, _ = img_rgb.shape
    if model is None:
        return img_rgb.copy()

    # torch.inference_mode(): disables autograd tracking for ~20-30% faster
    # LaMa inference and lower memory on both CPU and CUDA.
    with torch.inference_mode():
        # Reasonably sized image: one direct pass.
        if max(h, w) <= max_tile_size:
            pil_in = model(Image.fromarray(img_rgb), Image.fromarray(mask_gray))
            if pil_in.size != (w, h):
                pil_in = pil_in.resize((w, h), Image.Resampling.BILINEAR)
            return np.array(pil_in)

        mask_bin = (mask_gray > 10).astype(np.uint8)
        if np.sum(mask_bin) == 0:
            return img_rgb.copy()

        output_rgb = img_rgb.copy()

        # Cluster only nearby spots (+20px merge), not the whole face.
        merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        merged_clusters = cv2.dilate(mask_bin, merge_kernel)
        c_num, c_labels, c_stats, _ = cv2.connectedComponentsWithStats(merged_clusters)
        if c_num <= 1:
            return img_rgb.copy()

        for c in range(1, c_num):
            bx = int(c_stats[c, cv2.CC_STAT_LEFT])
            by = int(c_stats[c, cv2.CC_STAT_TOP])
            bw = int(c_stats[c, cv2.CC_STAT_WIDTH])
            bh = int(c_stats[c, cv2.CC_STAT_HEIGHT])

            # Adaptive context padding: scale by cluster size for optimal quality.
            # Small spots (3-8px radius): tight 60px context → faster, focused sampling.
            # Medium spots (8-15px): standard 80-120px context.
            # Large acne clusters (15-25px): wide 140-180px context → smooth transitions.
            cluster_radius = max(bw, bh) / 2.0
            adaptive_pad = int(np.clip(cluster_radius * 4.5, 60, 180))
            # User-provided context_pad acts as a baseline; adaptive scales from it.
            effective_pad = max(context_pad, adaptive_pad)

            x1 = max(0, bx - effective_pad)
            y1 = max(0, by - effective_pad)
            x2 = min(w, bx + bw + effective_pad)
            y2 = min(h, by + bh + effective_pad)

            crop_w = x2 - x1
            crop_h = y2 - y1

            crop_img = img_rgb[y1:y2, x1:x2]
            crop_mask = mask_gray[y1:y2, x1:x2]

            if np.sum(crop_mask > 0) == 0:
                continue

            tile_scale = min(1.0, float(max_tile_size) / float(max(crop_w, crop_h)))

            try:
                if tile_scale < 1.0:
                    small_w = max(64, int(round(crop_w * tile_scale)))
                    small_h = max(64, int(round(crop_h * tile_scale)))
                    small_img = cv2.resize(crop_img, (small_w, small_h), interpolation=cv2.INTER_AREA)
                    small_mask = cv2.resize(crop_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
                    small_mask = np.where(small_mask > 10, 255, 0).astype(np.uint8)
                    crop_pil = model(Image.fromarray(small_img), Image.fromarray(small_mask))
                    crop_inpainted = np.array(crop_pil)
                    if crop_inpainted.shape[:2] != (crop_h, crop_w):
                        crop_inpainted = cv2.resize(crop_inpainted, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)
                else:
                    crop_pil = model(Image.fromarray(crop_img), Image.fromarray(crop_mask))
                    crop_inpainted = np.array(crop_pil)
                    if crop_inpainted.shape[:2] != (crop_h, crop_w):
                        crop_inpainted = cv2.resize(crop_inpainted, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)
            except Exception as e:
                logger.warning(f"Tile inpainting exception on crop [{x1}:{x2}, {y1}:{y2}]: {e}")
                continue

            # Feathered blend: only the masked core takes inpainted pixels; the
            # surrounding context always keeps its native original pixels.
            crop_alpha = cv2.GaussianBlur((crop_mask > 0).astype(np.float32), (7, 7), 0)[:, :, None]
            output_rgb[y1:y2, x1:x2] = np.clip(
                crop_inpainted.astype(np.float32) * crop_alpha + output_rgb[y1:y2, x1:x2].astype(np.float32) * (1.0 - crop_alpha),
                0, 255
            ).astype(np.uint8)

    return output_rgb
