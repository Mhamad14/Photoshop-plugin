import cv2
import numpy as np
from PIL import Image

def auto_detect_blemishes(
    img_pil: Image.Image,
    sensitivity: float = 0.5,
    min_size: int = 4,
    max_size: int = 35,
    dilate_radius: int = 3
) -> Image.Image:
    """
    Face-Aware Skin Blemish & Acne Detector:
    1. Multi-Scale Difference of Gaussians (DoG) for circular spot detection
    2. Structural Edge & Facial Feature Exclusion (protects lips, nose contours, nostrils, eyes, and hair)
    3. Morphological roundness & aspect-ratio filtering (rejects lines, strands, and anatomical boundaries)
    4. Adaptive Z-score thresholding
    """
    img_rgb = np.array(img_pil.convert("RGB"))
    h, w, _ = img_rgb.shape
    
    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)
    
    # 1. Erythema (Redness) Index
    redness_index = (r - g) / (r + g + 10.0)
    
    # 2. Structural Edge Mask (Find nose outline, lips, chin boundary, hair)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 90)
    # Dilate edges to create an exclusion buffer
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edge_exclusion = cv2.dilate(edges, kernel_edge)
    
    # 3. Multi-Scale Difference of Gaussians (DoG) for isolated circular spots
    scales = [(1.2, 2.5), (2.0, 4.5), (3.5, 8.0), (6.0, 14.0)]
    dog_maps = []
    
    for s1, s2 in scales:
        g1 = cv2.GaussianBlur(redness_index, (0, 0), s1)
        g2 = cv2.GaussianBlur(redness_index, (0, 0), s2)
        dog = np.maximum(0, g1 - g2)
        dog_maps.append(dog)
        
    multi_scale_spots = np.maximum.reduce(dog_maps)
    
    # 4. Zero-out strong structural edges (nose ridge, jawline, hair boundaries)
    multi_scale_spots[edge_exclusion > 0] *= 0.1
    
    # 5. Adaptive Thresholding
    mean_val = np.mean(multi_scale_spots)
    std_val = np.std(multi_scale_spots)
    
    # Sensitivity controls threshold
    k = (1.0 - sensitivity) * 3.0 + 0.6
    threshold = mean_val + (k * std_val)
    
    binary_mask = (multi_scale_spots > threshold).astype(np.uint8) * 255
    
    # 6. Morphological Shape & Connected Components Filtering
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    filtered_mask = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width_i = stats[i, cv2.CC_STAT_WIDTH]
        height_i = stats[i, cv2.CC_STAT_HEIGHT]
        
        if width_i == 0 or height_i == 0:
            continue
            
        aspect_ratio = float(width_i) / float(height_i)
        
        # Pimples are relatively compact and circular (not long lines like hair or nose edge)
        if 0.3 <= aspect_ratio <= 3.0:
            # Check maximum diameter
            if min_size <= area <= (max_size * max_size):
                if width_i <= (max_size * 1.5) and height_i <= (max_size * 1.5):
                    filtered_mask[labels == i] = 255
                    
    # 7. Circular Dilation
    if dilate_radius > 0:
        kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilate_radius * 2 + 1, dilate_radius * 2 + 1)
        )
        filtered_mask = cv2.dilate(filtered_mask, kernel_dilate, iterations=1)
        
    return Image.fromarray(filtered_mask)
