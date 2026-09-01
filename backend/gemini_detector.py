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
        for candidate_model in ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite"]:
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


def analyze_gemini_pure_cloud(
    img_pil: Image.Image,
    api_key: Optional[str] = None,
    custom_instruction: Optional[str] = None
) -> Dict[str, Any]:
    """
    Pure Cloud Vision AI Analyzer:
    Analyzes the portrait using Gemini Vision model directly with zero heavy local dependencies.
    Provides detailed blemish localization, classification (pustule, papule, scar, spot),
    skin condition summary, clarity index, and recommended retouch strategy.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return {
            "success": False,
            "error": "No Gemini API Key configured. Please provide an API key in Studio Settings."
        }

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        w, h = img_pil.size
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

        instruction_context = f"\nUser Directive: {custom_instruction}" if custom_instruction else ""

        prompt = f"""
You are an expert dermatological AI vision system.
Analyze the facial skin in this portrait with high precision.
{instruction_context}

Identify ALL visible skin blemishes:
1. "inflamed_pimple" (red inflammatory papules/pustules)
2. "whitehead" / "blackhead" (comedones)
3. "dark_spot" (hyperpigmentation / post-inflammatory erythema)
4. "acne_scar" (focal micro scars)
5. "ingrown_hair" / "razor_bump"

DO NOT mark:
- Eyes, lips, nostrils, teeth, eyebrows, hair, natural beauty marks/freckles (unless requested).
- Normal clean skin pores.

Return strictly a JSON object with:
{{
  "overall_skin_clarity": 82, // integer 0-100
  "skin_type_detected": "Combination / Acne-Prone",
  "dermatological_summary": "Brief 1-2 sentence medical-grade skin assessment.",
  "recommended_actions": ["Action 1", "Action 2"],
  "zone_analysis": {{
    "forehead": "Clear / Mild breakouts",
    "cheeks": "Moderate inflammatory papules",
    "chin_jaw": "Hormonal acne cluster",
    "nose": "Mild sebaceous filaments"
  }},
  "blemishes": [
    {{
      "box_2d": [ymin, xmin, ymax, xmax], // 0-1000 normalized or 0.0-1.0
      "type": "inflamed_pimple",
      "severity": "mild" | "moderate" | "severe",
      "location_desc": "Right cheek lower quadrant",
      "confidence": 0.96
    }}
  ]
}}
"""

        response = None
        used_model = "unknown"
        for candidate_model in ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-2.0-flash"]:
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
                    used_model = candidate_model
                    break
            except Exception as e:
                logger.warning(f"Cloud candidate model {candidate_model} failed: {e}")

        if not response or not response.text:
            return {"success": False, "error": "Gemini API did not return a response."}

        parsed_data = json.loads(response.text.strip())
        raw_blemishes = parsed_data.get("blemishes", [])

        # Build coordinate boxes & canvas overlay blobs
        blobs = []
        blob_id = 1
        for b in raw_blemishes:
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

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            radius = max(5, int(math.hypot(bw, bh) / 2.0))

            blobs.append({
                "id": blob_id,
                "bbox": [x1, y1, x2, y2],
                "centroid": [round(cx, 1), round(cy, 1)],
                "radius": radius,
                "area": int(math.pi * radius * radius),
                "confidence": float(b.get("confidence", 0.95)),
                "active": True,
                "label": b.get("type", "inflamed_pimple"),
                "severity": b.get("severity", "moderate"),
                "location_desc": b.get("location_desc", f"Spot #{blob_id}"),
                "source": "gemini_cloud_api"
            })
            blob_id += 1

        return {
            "success": True,
            "engine": "Gemini Vision AI Cloud",
            "model_used": used_model,
            "clarity_score": parsed_data.get("overall_skin_clarity", 85),
            "skin_type": parsed_data.get("skin_type_detected", "Acne-prone / Sensitive"),
            "summary": parsed_data.get("dermatological_summary", "Dermatological analysis complete."),
            "recommendations": parsed_data.get("recommended_actions", ["Perform AI healing on active papules"]),
            "zone_analysis": parsed_data.get("zone_analysis", {}),
            "blobs": blobs,
            "total_detected": len(blobs)
        }

    except Exception as e:
        logger.error(f"Cloud Gemini analysis error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

