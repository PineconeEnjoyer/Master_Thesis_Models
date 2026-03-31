import os
import cv2
from tqdm import tqdm

def remove_hair(image_path, save_path):
    # Wczytanie obrazu przez OpenCV
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print(f"Nie udało się wczytać: {image_path}")
        return

    # Skala szarości
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Filtracja morfologiczna (Black-Hat)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Binarna maska
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Inpainting
    img_inpainted = cv2.inpaint(img_cv, mask, 1, cv2.INPAINT_TELEA)

    # Zapis obrazu
    cv2.imwrite(save_path, img_inpainted)


def process_entire_dataset(raw_dir, clean_dir):
    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)

    image_files = [f for f in os.listdir(raw_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Rozpoczynam czyszczenie {len(image_files)} zdjęć...")

    for filename in tqdm(image_files, desc="Usuwanie włosów"):
        raw_path = os.path.join(raw_dir, filename)
        clean_path = os.path.join(clean_dir, filename)

        # Pomijamy, jeśli plik już został wcześniej przetworzony
        if not os.path.exists(clean_path):
            remove_hair(raw_path, clean_path)


if __name__ == "__main__":
    RAW_IMAGE_DIR = 'HAM10000_images'  # Folder z oryginalnymi zdjęciami
    CLEAN_IMAGE_DIR = 'HAM10000_images_clean'  # Folder docelowy na wyczyszczone zdjęcia

    process_entire_dataset(RAW_IMAGE_DIR, CLEAN_IMAGE_DIR)
    print("Preprocessing zakończony.")