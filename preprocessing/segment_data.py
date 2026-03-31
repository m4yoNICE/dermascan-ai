import cv2
import numpy as np
import os  # 👈 add this

def segment_skin_color(image_path):
    # 👇 Add this block at the very start of the function
    if not os.path.exists(image_path):
        print(f"❌ File NOT found: {image_path}")
        return None, None
    else:
        print(f"✅ File found: {image_path}")

    img = cv2.imread(image_path)

    if img is None:
        print("❌ OpenCV could not read the file. Try converting it to .jpg")
        return None, None
    img = cv2.imread(image_path)

    # Convert to HSV and YCrCb color spaces
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Define skin color ranges in HSV
    lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
    upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Define skin color ranges in YCrCb
    lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # Combine both masks
    skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    # Clean up the mask (remove noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

    # Apply mask to original image
    skin_only = cv2.bitwise_and(img, img, mask=skin_mask)

    return skin_only, skin_mask


#Usage
result, mask = segment_skin_color(r"D:\Navigation\Downloads\SKIN-AUG\IMG20250915165343.jpg")
cv2.imwrite("skin_segmented.jpg", result)