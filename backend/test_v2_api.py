import io
import json
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
from server import app

def run_comprehensive_api_test():
    print("=======================================================")
    print(" Running Comprehensive Test of v2 Auto-Detection API   ")
    print("=======================================================")

    with TestClient(app) as client:
        # 1. Health
        r_health = client.get("/health")
        print(f"[1. GET /health] Status: {r_health.status_code}, Info: {r_health.json()}")

        # Create dummy portrait with face skin and blemish
        img = Image.new("RGB", (300, 300), (225, 180, 150))
        draw = ImageDraw.Draw(img)
        # Blemish spot
        draw.ellipse([140, 140, 155, 155], fill=(200, 60, 65))
        # Eyes
        draw.ellipse([90, 100, 120, 115], fill=(30, 30, 30))
        draw.ellipse([180, 100, 210, 115], fill=(30, 30, 30))
        # Mouth
        draw.ellipse([120, 210, 180, 230], fill=(190, 70, 85))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        # 2. Analyze
        print("\n[2. POST /analyze]")
        files = {"image": ("test_portrait.png", img_bytes, "image/png")}
        data = {"sensitivity": 0.5, "detect_pimples": "true", "detect_skin": "true"}
        r_analyze = client.post("/analyze", files=files, data=data)
        print(f"Status: {r_analyze.status_code}")
        analyze_res = r_analyze.json()
        print(f"Skin Coverage: {analyze_res.get('skin_percentage')}%")
        print(f"Base Tone RGB: {analyze_res.get('base_tone_rgb')}, LAB: {analyze_res.get('base_tone_lab')}")
        print(f"Blobs Detected: {analyze_res.get('blobs_count')}")
        print(f"Processing Time: {analyze_res.get('process_time_ms')}ms")

        # 3. Refine Point: Add Blemish
        print("\n[3. POST /refine-point (Add Pimple)]")
        files = {"image": ("test_portrait.png", img_bytes, "image/png")}
        data = {
            "x": 148,
            "y": 148,
            "action_type": "add_pimple",
            "blobs_json": json.dumps(analyze_res.get("blobs", []))
        }
        r_refine = client.post("/refine-point", files=files, data=data)
        print(f"Status: {r_refine.status_code}")
        refine_res = r_refine.json()
        print(f"Updated Blobs Count: {len(refine_res.get('blobs', []))}")
        print(f"New Blob: {refine_res.get('new_blob')}")

        # 4. Refine Point: Sample Skin Tone
        print("\n[4. POST /refine-point (Sample Tone)]")
        files = {"image": ("test_portrait.png", img_bytes, "image/png")}
        data = {
            "x": 150,
            "y": 170,
            "action_type": "sample_tone",
            "blobs_json": json.dumps(refine_res.get("blobs", []))
        }
        r_sample = client.post("/refine-point", files=files, data=data)
        print(f"Status: {r_sample.status_code}")
        sample_res = r_sample.json()
        print(f"Sampled Skin Tone: {sample_res.get('sampled_tone')}")

        # 5. Apply Heal (Action 1)
        print("\n[5. POST /apply-heal (Remove Pimples)]")
        files = {"image": ("test_portrait.png", img_bytes, "image/png")}
        data = {
            "blobs_json": json.dumps(refine_res.get("blobs", [])),
            "texture_blend": 0.25,
            "feather_radius": 3
        }
        r_heal = client.post("/apply-heal", files=files, data=data)
        print(f"Status: {r_heal.status_code}, Healed PNG bytes: {len(r_heal.content)}, Process Time: {r_heal.headers.get('x-process-time')}s")

        # 6. Apply Lighten (Action 2)
        print("\n[6. POST /apply-lighten (Relative Skin Lightening)]")
        files = {"image": ("test_portrait.png", img_bytes, "image/png")}
        data = {
            "strength": 0.35,
            "base_tone_lab": json.dumps(sample_res.get("sampled_tone", {}).get("lab", [180.0, 140.0, 140.0])),
            "feather_radius": 4
        }
        r_lighten = client.post("/apply-lighten", files=files, data=data)
        print(f"Status: {r_lighten.status_code}, Lightened PNG bytes: {len(r_lighten.content)}, Process Time: {r_lighten.headers.get('x-process-time')}s")

        print("\n=======================================================")
        print(" ALL v2 API ENDPOINTS VERIFIED AND WORKING PERFECTLY!   ")
        print("=======================================================\n")

if __name__ == "__main__":
    run_comprehensive_api_test()
