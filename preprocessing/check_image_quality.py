import cv2
import numpy as np

MAX_IMAGE_SIZE_MB = 10
MAX_IMAGE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
BLUR_THRESHOLD = 10.0

def check_image_quality(image_data: bytes) -> dict:
    if len(image_data) > MAX_IMAGE_BYTES:
        return {"ok": False, "reason": "too_large"}

    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "reason": "invalid_format"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_intensity = float(gray.mean())
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    h, w = gray.shape

    if mean_intensity < 20:
        return {"ok": False, "reason": "too_dark", "brightness": mean_intensity}
    if mean_intensity > 240:
        return {"ok": False, "reason": "too_bright", "brightness": mean_intensity}
    if variance < BLUR_THRESHOLD:
        return {"ok": False, "reason": "too_blurry", "variance": variance}

    return {"ok": True, "variance": variance, "brightness": mean_intensity,
            "resolution": {"width": w, "height": h}}