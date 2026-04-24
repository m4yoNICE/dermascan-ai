import cv2
import numpy as np

def is_skin(image_bytes: bytes, threshold: float = 0.10) -> bool:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 20, 50], dtype=np.uint8)
    upper1 = np.array([25, 180, 255], dtype=np.uint8)
    lower2 = np.array([160, 20, 50], dtype=np.uint8)
    upper2 = np.array([180, 180, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    ratio = np.sum(mask > 0) / mask.size
    print(f"  Skin ratio: {ratio:.2%}")
    return ratio >= threshold