import io
import json
import os
import logging
from typing import Optional, List, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger("gemini_detector")

def detect_blemishes_gemini(
    img_pil: Image.Image,
    api_key: Optional[str] = None,
    dilate_pixels: int = 4
) -> Optional[Image.Image]:
    """
    Uses Gemini Vision AI to detect all acne, pimples, pustules, and blemishes
    with human-level dermatological precision, completely ignoring lips, nose, eyes, and hair.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        logger.warning("No GEMINI_API_KEY provided.")
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)

        # Convert image to bytes
        img_buf = io.BytesIO()
        img_pil.convert("RGB").save(img_buf, format="JPEG", quality=90)
        img_bytes = img_buf.getvalue()

        prompt = """
You are a precision retouching vision AI.
Detect ALL skin imperfections that a professional photo retoucher would remove, in this portrait photo:
acne, pimples, pustules, whiteheads, blackheads, cysts, milia, red inflamed patches, acne scars,
dark spots, sun spots, age spots, and small flat blemishes.

CRITICAL INSTRUCTIONS:
1. ONLY detect actual skin imperfections a retoucher would hide.
2. DO NOT detect: lips, mouth, nose tip, nostrils, eyes, eyebrows, ears, hair, teeth, jewelry, clothing.
3. DO NOT detect: large identity moles or beauty marks, freckle patterns spread evenly across large areas, or natural smile lines.
4. Each detection must be a small localized region (roughly under 2% of the face area).
5. For each detected imperfection, return its precise 2D bounding box in normalized coordinates [ymin, xmin, ymax, xmax] on a scale of 0 to 1000, plus a short label.

Output strictly valid JSON with this schema:
{
  "blemishes": [
    {"box_2d": [ymin, xmin, ymax, xmax], "label": "pimple|blackhead|scar|dark_spot|red_patch|milia"}
  ]
}
"""

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )

        text = response.text.strip()
        logger.info(f"Gemini Vision response: {text[:200]}...")

        data = json.loads(text)
        blemishes = data.get("blemishes", [])
        if not blemishes and isinstance(data, list):
            blemishes = data

        w, h = img_pil.size
        mask = Image.new("L", (w, h), color=0)
        draw = ImageDraw.Draw(mask)

        logger.info(f"Gemini detected {len(blemishes)} distinct blemishes on the skin.")

        for b in blemishes:
            box = b.get("box_2d") or b.get("box") or b.get("bbox")
            if not box or len(box) != 4:
                continue

            ymin, xmin, ymax, xmax = box
            
            # Normalize scale (0-1000 to pixels)
            # If coordinates are 0-1 float:
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

            # Add padding margin
            x1 = max(0, x1 - dilate_pixels)
            y1 = max(0, y1 - dilate_pixels)
            x2 = min(w, x2 + dilate_pixels)
            y2 = min(h, y2 + dilate_pixels)

            # Draw filled ellipse over each blemish
            draw.ellipse([x1, y1, x2, y2], fill=255)

        # Smooth edges slightly
        mask_smooth = mask.filter(ImageFilter.GaussianBlur(radius=2))
        return mask_smooth

    except Exception as e:
        logger.error(f"Gemini Vision detection failed: {e}", exc_info=True)
        return None
