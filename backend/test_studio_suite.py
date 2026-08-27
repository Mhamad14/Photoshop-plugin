import io
import time
import numpy as np
from PIL import Image, ImageDraw

def test_all_modules():
    print("=== 1. Testing AI Dodge & Burn Engine ===")
    from dodge_and_burn import generate_dodge_and_burn_map, create_dodge_and_burn_rgba_patch
    img = np.full((256, 256, 3), 200, dtype=np.uint8)
    # Add artificial shadow
    img[100:150, 100:150] = 130
    mask = np.full((256, 256), 255, dtype=np.uint8)
    comp, gray, alpha = generate_dodge_and_burn_map(img, mask, strength=0.6)
    rgba = create_dodge_and_burn_rgba_patch(img, mask)
    assert comp.shape == (256, 256, 3), "D&B comp shape mismatch"
    assert gray.shape == (256, 256), "D&B gray map mismatch"
    assert rgba.size == (256, 256), "D&B rgba size mismatch"
    print("   [+] Dodge & Burn passed!")

    print("=== 2. Testing AI Shine & Glare Neutralizer ===")
    from shine_neutralizer import neutralize_skin_shine, create_shine_neutralizer_rgba_patch
    # Add shiny hot-spot
    img_shine = img.copy()
    img_shine[50:80, 50:80] = 255
    comp_shine, alpha_shine = neutralize_skin_shine(img_shine, mask, strength=0.5)
    rgba_shine = create_shine_neutralizer_rgba_patch(img_shine, mask)
    assert comp_shine.shape == (256, 256, 3), "Shine comp shape mismatch"
    assert rgba_shine.size == (256, 256), "Shine rgba size mismatch"
    print("   [+] Shine Neutralizer passed!")

    print("=== 3. Testing Dermatological Spot Classifier ===")
    from spot_classifier import classify_spot, filter_blobs_by_preference
    # Create test image with realistic skin background, red acne spot and dark mole spot
    test_img = np.full((200, 200, 3), [220, 175, 145], dtype=np.uint8)
    test_img[30:50, 30:50] = [235, 60, 60]   # Red inflamed pimple
    test_img[100:120, 100:120] = [70, 50, 45] # Dark melanin mole
    base_lab = np.array([178.0, 140.0, 143.0])

    pimple_class = classify_spot(test_img, 40, 40, 8, base_lab)
    mole_class = classify_spot(test_img, 110, 110, 8, base_lab)
    print(f"   [+] Spot 1 classification: {pimple_class['type']} (is_mole={pimple_class['is_mole']})")
    print(f"   [+] Spot 2 classification: {mole_class['type']} (is_mole={mole_class['is_mole']})")

    blobs = [
        {"x": 40, "y": 40, "radius": 8, "active": True},
        {"x": 110, "y": 110, "radius": 8, "active": True}
    ]
    filtered_blobs = filter_blobs_by_preference(blobs, test_img, preserve_moles=True)
    assert len(filtered_blobs) == 1, "Mole filtering failed"
    print(f"   [+] Preserved mole! Remaining active spots: {len(filtered_blobs)}")

    print("=== 4. Testing FastAPI App Endpoints via TestClient ===")
    from fastapi.testclient import TestClient
    from server import app
    client = TestClient(app)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    print(f"   [+] /health: version {health_data['version']}, tools: {len(health_data['tools'])}")

    # Test /apply-dodge-burn
    pil_buf = io.BytesIO()
    Image.fromarray(img).save(pil_buf, format="PNG")
    files = {"image": ("test.png", pil_buf.getvalue(), "image/png")}
    db_resp = client.post("/apply-dodge-burn", files=files, data={"strength": 0.5})
    assert db_resp.status_code == 200, f"/apply-dodge-burn failed: {db_resp.text}"
    print(f"   [+] /apply-dodge-burn endpoint succeeded ({len(db_resp.content)} bytes)")

    # Test /apply-shine-neutralize
    files = {"image": ("test.png", pil_buf.getvalue(), "image/png")}
    shine_resp = client.post("/apply-shine-neutralize", files=files, data={"strength": 0.5})
    assert shine_resp.status_code == 200, f"/apply-shine-neutralize failed: {shine_resp.text}"
    print(f"   [+] /apply-shine-neutralize endpoint succeeded ({len(shine_resp.content)} bytes)")

    print("\nALL STUDIO TOOLS & ENDPOINTS VERIFIED AND PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_all_modules()
