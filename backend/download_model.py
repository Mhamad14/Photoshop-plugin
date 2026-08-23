import os
import sys
import requests
from tqdm import tqdm

def download_weights():
    cache_dir = os.path.expanduser(r"~\.cache\torch\hub\checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    target_file = os.path.join(cache_dir, "big-lama.pt")

    if os.path.exists(target_file) and os.path.getsize(target_file) > 150_000_000:
        print(f"[+] Model file already exists at {target_file} ({os.path.getsize(target_file)} bytes)")
        return True

    # Hugging Face CDN mirror
    url = "https://huggingface.co/fashn-ai/LaMa/resolve/main/big-lama.pt"
    print(f"[*] Downloading Simple-LaMa model weights from CDN: {url}")
    print(f"[*] Target destination: {target_file}")

    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()

    total_size = int(resp.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1MB chunks

    with open(target_file, "wb") as f, tqdm(
        desc="big-lama.pt",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=block_size):
            size = f.write(data)
            bar.update(size)

    print(f"[+] Download complete: {target_file} ({os.path.getsize(target_file)} bytes)")
    return True

if __name__ == "__main__":
    download_weights()
