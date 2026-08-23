import io
import time
import requests
from PIL import Image, ImageDraw

SERVER_URL = "http://127.0.0.1:8001"

def find_server_url():
    global SERVER_URL
    candidates = ["http://127.0.0.1:8001", "http://127.0.0.1:8008", "http://127.0.0.1:8000"]
    for url in candidates:
        try:
            resp = requests.get(f"{url}/health", timeout=2)
            if resp.status_code == 200:
                SERVER_URL = url
                return url
        except Exception:
            pass
    return SERVER_URL

def test_auto_heal():
    url = find_server_url()
    print(f"[*] Testing /auto-heal on {url} ...")
    
    # Create synthetic skin portrait with red blemish
    img = Image.new("RGB", (250, 250), color=(225, 175, 145))
    draw = ImageDraw.Draw(img)
    # Draw red acne spots
    draw.ellipse([50, 50, 70, 70], fill=(210, 60, 60))
    draw.ellipse([150, 140, 175, 165], fill=(200, 50, 50))
    draw.ellipse([100, 180, 115, 195], fill=(190, 70, 70))
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    
    files = {"image": ("portrait.png", img_bytes.getvalue(), "image/png")}
    data = {"sensitivity": 0.5, "texture_blend": 0.25, "feather_radius": 3}
    
    t0 = time.time()
    resp = requests.post(f"{url}/auto-heal", files=files, data=data, timeout=30)
    elapsed = time.time() - t0
    
    if resp.status_code == 200:
        print(f"[+] /auto-heal succeeded in {elapsed:.3f}s! Output size: {len(resp.content)} bytes")
        out_img = Image.open(io.BytesIO(resp.content))
        print(f"[+] Output mode: {out_img.mode}, size: {out_img.size}")
        return True
    else:
        print(f"[-] /auto-heal failed ({resp.status_code}): {resp.text}")
        return False

if __name__ == "__main__":
    print("=== Testing Auto-Heal AI ===")
    test_auto_heal()
