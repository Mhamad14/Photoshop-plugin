import io
import json
import os
import math
import logging
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger("gemini_detector")

def detect_blemishes_gemini_blobs(
    img_pil: Image.Image,
    api_key: Optional[str] = None,
    dilate_pixels: int = 2
) -> Tuple[List[Dict[str, Any]], Optional[Image.Image]]:
    """
    Uses Gemini Vision AI to detect acne, pimples, pustules, and focal blemishes
    with dermatological precision. Returns a list of structured blobs and a binary mask.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        logger.warning("No GEMINI_API_KEY provided.")
        return [], None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)

        w, h = img_pil.size
        # Resize to max 1280 for fast VLM inference if image is huge
        scale = min(1.0, 1280.0 / float(max(w, h)))
        if scale < 1.0:
            target_w = max(1, int(w * scale))
            target_h = max(1, int(h * scale))
            resized_pil = img_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        else:
            resized_pil = img_pil

        img_buf = io.BytesIO()
        resized_pil.convert("RGB").save(img_buf, format="JPEG", quality=90)
        img_bytes = img_buf.getvalue()

        prompt = """
You are a master dermatological retouching AI.
Detect ONLY distinct, localized skin blemishes in this portrait:
- Inflamed pimples, pustules, whiteheads, blackheads, and red acne bumps.
- Small focal acne scars and dark hyperpigmentation spots.

STRICT NEGATIVE CONSTRAINTS:
1. DO NOT detect lips, nostrils, nose bridge, eyelids, eyelashes, eyebrows, teeth, chin folds, neck creases, or hair.
2. DO NOT detect large patches of normal skin or entire cheeks. Every blemish must be a small discrete spot (under 1.5% of the face).
3. DO NOT detect natural beauty marks/moles unless they are inflamed acne lesions.
4. Limit detections to true focal blemishes that need healing.

Output JSON format:
{
  "blemishes": [
    {"box_2d": [ymin, xmin, ymax, xmax], "label": "pimple"}
  ]
}
"""

        response = None
        last_err = None
        for candidate_model in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-pro"]:
            try:
                response = client.models.generate_content(
                    model=candidate_model,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        prompt
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1
                    )
                )
                if response and response.text:
                    logger.info(f"Gemini Vision detection succeeded with model: {candidate_model}")
                    break
            except Exception as model_err:
                last_err = model_err
                logger.warning(f"Candidate model {candidate_model} failed: {model_err}")

        if response is None or not response.text:
            raise last_err or Exception("All Gemini candidate models failed.")

        text = response.text.strip()
        data = json.loads(text)
        blemishes = data.get("blemishes", [])
        if not blemishes and isinstance(data, list):
            blemishes = data

        mask = Image.new("L", (w, h), color=0)
        draw = ImageDraw.Draw(mask)
        blobs: List[Dict[str, Any]] = []

        logger.info(f"Gemini detected {len(blemishes)} distinct blemishes.")

        blob_id = 1
        for b in blemishes:
            box = b.get("box_2d") or b.get("box") or b.get("bbox")
            if not box or len(box) != 4:
                continue

            ymin, xmin, ymax, xmax = box
            
            if all(0.0 <= coord <= 1.0 for coord in box):
                y1 = int(ymin * h)
                x1 = int(xmin * w)
                y2 = int(ymax * h)
                x2 = int(xmax * w)
            else:
                y1 = int((ymin / 1000.0) * h)
                x1 = int((xmin / 1000.0) * w)
                y2 = int((ymax / 1000.0) * h)
                x2 = int((xmax / 1000.0) * w)

            # Clamp coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            # Prevent giant boxes from erasing whole facial regions
            max_allowed_dim = int(min(w, h) * 0.08)
            if bw > max_allowed_dim or bh > max_allowed_dim:
                # Center small radius
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                radius = max_allowed_dim // 2
                x1 = int(max(0, cx - radius))
                y1 = int(max(0, cy - radius))
                x2 = int(min(w, cx + radius))
                y2 = int(min(h, cy + radius))

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            radius = max(4, int(math.hypot(x2 - x1, y2 - y1) / 2.2))

            draw.ellipse([x1, y1, x2, y2], fill=255)

            blobs.append({
                "id": blob_id,
                "bbox": [x1, y1, x2, y2],
                "centroid": [round(cx, 1), round(cy, 1)],
                "radius": radius,
                "area": int(math.pi * radius * radius),
                "confidence": 0.95,
                "active": True,
                "label": b.get("label", "pimple"),
                "source": "vlm_gemini"
            })
            blob_id += 1

        mask_smooth = mask.filter(ImageFilter.GaussianBlur(radius=1))
        return blobs, mask_smooth

    except Exception as e:
        logger.error(f"Gemini Vision detection failed: {e}", exc_info=True)
        return [], None


def detect_blemishes_gemini(
    img_pil: Image.Image,
    api_key: Optional[str] = None,
    dilate_pixels: int = 2
) -> Optional[Image.Image]:
    blobs, mask = detect_blemishes_gemini_blobs(img_pil, api_key, dilate_pixels)
    return mask
