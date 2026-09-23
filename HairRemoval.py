import os
import cv2
import numpy as np
from tqdm import tqdm

def create_hair_mask(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    local_bg = cv2.medianBlur(gray, 35)
    local_darkness = cv2.subtract(local_bg, gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    combined_features = cv2.bitwise_and(blackhat, local_darkness)
    blurred = cv2.GaussianBlur(combined_features, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 8, 255, cv2.THRESH_BINARY)

    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)

    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        longer_side = max(w, h)
        area = cv2.contourArea(contour)

        if area >= 3 and longer_side >= 15:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

    filtered_mask = cv2.dilate(filtered_mask, morph_kernel, iterations=1)

    return filtered_mask


def remove_hair(image_path, save_path):
    img_cv = cv2.imread(image_path)

    if img_cv is None:
        print(f"Couldn't load the file: {image_path}")
        return

    mask = create_hair_mask(img_cv)

    img_inpainted = cv2.inpaint(
        img_cv,
        mask,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA
    )

    soft_mask = cv2.GaussianBlur(mask, (3, 3), 0)
    alpha = soft_mask.astype(np.float32) / 255.0
    alpha = alpha[:, :, None]

    img_clean = img_cv.astype(np.float32) * (1.0 - alpha) + img_inpainted.astype(np.float32) * alpha
    img_clean = np.clip(img_clean, 0, 255).astype(np.uint8)

    cv2.imwrite(save_path, img_clean)


def process_entire_dataset(raw_dir, clean_dir):
    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)

    image_files = [
        f for f in os.listdir(raw_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Starting cleaning {len(image_files)} images...")

    for filename in tqdm(image_files, desc="Hair removal"):
        raw_path = os.path.join(raw_dir, filename)
        clean_path = os.path.join(clean_dir, filename)

        if not os.path.exists(clean_path):
            remove_hair(raw_path, clean_path)


if __name__ == "__main__":
    RAW_IMAGE_DIR = r'HAM10000\HAM10000_images'
    CLEAN_IMAGE_DIR = r'HAM10000\HAM10000_images_mask'

    process_entire_dataset(RAW_IMAGE_DIR, CLEAN_IMAGE_DIR)
    print("Preprocessing finished.")