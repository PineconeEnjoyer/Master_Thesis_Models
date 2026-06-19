import os
import cv2
import numpy as np
from tqdm import tqdm

# Funkcja do stworzenia maski na włosy ze szczególną ostrożnością by nie usuwać wszystkich ciemnych plamek
def create_hair_mask(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Rozmycie medianowe o bardzo dużym oknie (np. 35) zignoruje cienkie włosy, ale zachowa ogólny kolor i gradienty samej zmiany skórnej
    local_bg = cv2.medianBlur(gray, 35)

    # Obliczamy, jak bardzo dany piksel jest ciemniejszy od swojej powierzchni
    # Włosy na zmianie będą miały tu wysoką wartość, a sama zmiana niską (bo jej powierzchnia też jest ciemna)
    local_darkness = cv2.subtract(local_bg, gray)

    # Filtr kształty, wciąż szukamy podłużnych, cienkich prążków
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Piksel przejdzie dalej TYLKO jeśli ma kształt włosa ORAZ jest ciemniejszy od swojej powierzchni
    combined_features = cv2.bitwise_and(blackhat, local_darkness)

    # Oczyszczanie i progowanie
    blurred = cv2.GaussianBlur(combined_features, (3, 3), 0)

    # Dzięki precyzyjniejszym cechom, próg może być bardziej stanowczy
    _, mask = cv2.threshold(blurred, 8, 255, cv2.THRESH_BINARY)

    # Zamykanie przerw
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel)

    # Filtracja wielkości i kształtu
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)

    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        longer_side = max(w, h)
        area = cv2.contourArea(contour)

        # Włosy wciąż muszą być w miarę długie, aby odsiać ewentualny szum i siateczkę pigmentową
        if area >= 3 and longer_side >= 15:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Poszerzenie dla inpaintingu
    filtered_mask = cv2.dilate(filtered_mask, morph_kernel, iterations=1)

    return filtered_mask


# Funkcja
def remove_hair(image_path, save_path):
    # Wczytanie obrazu przez OpenCV
    img_cv = cv2.imread(image_path)

    if img_cv is None:
        print(f"Nie udało się wczytać: {image_path}")
        return

    # Wygenerowanie maski binarnej
    mask = create_hair_mask(img_cv)

    # Inpainting na podstawie stworzonej maski i wgranego zdjęcia
    img_inpainted = cv2.inpaint(
        img_cv,
        mask,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA
    )

    # Zmiękczenie maski, żeby strefa zmiany skórnej była mniej narażona na inpainting
    soft_mask = cv2.GaussianBlur(mask, (3, 3), 0)
    alpha = soft_mask.astype(np.float32) / 255.0
    alpha = alpha[:, :, None]

    img_clean = img_cv.astype(np.float32) * (1.0 - alpha) + img_inpainted.astype(np.float32) * alpha
    img_clean = np.clip(img_clean, 0, 255).astype(np.uint8)

    # Zapisanie wyczyszczonego obrazu
    cv2.imwrite(save_path, img_clean)


# Funkcja pozwalająca przetworzyć większość ilość obrazów w jednym cyklu
def process_entire_dataset(raw_dir, clean_dir):
    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)

    image_files = [
        f for f in os.listdir(raw_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Rozpoczynam czyszczenie {len(image_files)} zdjęć...")

    # Wizualizacja procesu! :D
    for filename in tqdm(image_files, desc="Usuwanie włosów"):
        raw_path = os.path.join(raw_dir, filename)
        clean_path = os.path.join(clean_dir, filename)

        # Pomijamy, jeśli plik już został wcześniej przetworzony
        if not os.path.exists(clean_path):
            remove_hair(raw_path, clean_path)


if __name__ == "__main__":
    RAW_IMAGE_DIR = r'HAM10000\HAM10000_images'
    CLEAN_IMAGE_DIR = r'HAM10000\HAM10000_images_mask'

    process_entire_dataset(RAW_IMAGE_DIR, CLEAN_IMAGE_DIR)
    print("Preprocessing zakończony.")