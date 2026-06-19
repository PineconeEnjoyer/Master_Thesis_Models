import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# Ustawienia i transformacje augmentacyjne do zwiększenia różorodości danych
IMG_SIZE = 224

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    #transforms.RandomRotation(degrees=45),
    transforms.RandomAffine(degrees=90, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Deklaracja datasetu, ścierzek oraz narzędzi używanych w kodzie
class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, image_dir, mask_dir, transform=None, classes=None, seg_mode='crop'):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.seg_mode = seg_mode

        if classes is None:
            self.classes = sorted(self.dataframe['dx'].unique())
        else:
            self.classes = list(classes)

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_id = self.dataframe.iloc[idx]['image_id']

        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{image_id}_segmentation.png")

        try:
            image = Image.open(img_path).convert('RGB')
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                image = apply_segmentation(image, mask, mode=self.seg_mode)
            else:
                print(f"Ostrzeżenie: Brak maski dla {image_id}. Przetwarzam bez segmentacji.")

        except FileNotFoundError:
            print(f"Ostrzeżenie: Brak pliku obrazu {img_path}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))

        label_name = self.dataframe.iloc[idx]['dx']
        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label


# Segmentacja obrazu poprzez nałożenie maski, żeby Resize nie uciął istotnych informacji
def apply_segmentation(image_pil, mask_pil, mode='crop', margin=15):
    # Konwersja PIL do formatu OpenCV, potrzebna jednokanałowość
    img = np.array(image_pil)
    mask = np.array(mask_pil.convert('L'))

    # Progowanie maski na wypadek, gdyby nie była idealnie binarna
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Zależnie od trybu, wyczerniamy tło z binarnej maski lub szukamy konturów na jego podstawie
    if mode == 'mask':
        mask_normalized = (mask_binary / 255.0).astype(np.uint8)
        mask_rgb = np.stack([mask_normalized] * 3, axis=-1)

        segmented_img = img * mask_rgb
        return Image.fromarray(segmented_img)

    elif mode == 'crop':
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image_pil

        # Znalezienie największego konturu
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Dodanie marginesu, pilnując by nie wyjść poza obraz
        h_img, w_img, _ = img.shape
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(w_img, x + w + margin)
        y_end = min(h_img, y + h + margin)

        cropped_img = img[y_start:y_end, x_start:x_end]
        return Image.fromarray(cropped_img)


# Dzielenie danych na odpowiednie grupy by zapobiec wyciekom przy trenowaniu modelu
def create_stratified_group_splits(metadata_path, test_size=0.15, val_size=0.15):
    df = pd.read_csv(metadata_path)

    # Stratyfikowany podział po lesion_id
    lesion_df = df.groupby('lesion_id')['dx'].first().reset_index()
    temp_size = test_size + val_size

    # Podział na grupę treningową i resztę
    train_lesions, temp_lesions = train_test_split(
        lesion_df, test_size=temp_size, stratify=lesion_df['dx'], random_state=42
    )

    # Podział na grupę testową i validacyjną
    val_ratio = val_size / temp_size
    val_lesions, test_lesions = train_test_split(
        temp_lesions, test_size=(1-val_ratio), stratify=temp_lesions['dx'], random_state=42
    )

    train_df = df[df['lesion_id'].isin(train_lesions['lesion_id'])].reset_index(drop=True)
    val_df = df[df['lesion_id'].isin(val_lesions['lesion_id'])].reset_index(drop=True)
    test_df = df[df['lesion_id'].isin(test_lesions['lesion_id'])].reset_index(drop=True)

    return train_df, val_df, test_df


# Funkcja przygotowująca DataLoadery, wagi oraz klasy do przekazania modelom
def get_dataloaders_and_weights(metadata_path, image_dir, mask_dir, batch_size=32, num_workers=2):
    print("Wczytywanie i podział metadanych...")
    train_df, val_df, test_df = create_stratified_group_splits(metadata_path, test_size=0.15, val_size=0.15)

    # Wyliczanie klas i wag klas na podstawie zbioru treningowego
    classes = np.array(sorted(train_df['dx'].unique()))
    y_train = train_df['dx'].values

    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weights = torch.tensor(weights, dtype=torch.float)

    # Inicjalizacja Datasetów
    print("Inicjalizacja zbiorów danych...")
    train_dataset = SkinLesionDataset(train_df, image_dir, mask_dir, transform=train_transforms, classes=classes)
    val_dataset = SkinLesionDataset(val_df, image_dir, mask_dir, transform=val_test_transforms, classes=classes)
    test_dataset = SkinLesionDataset(test_df, image_dir, mask_dir, transform=val_test_transforms, classes=classes)

    # Inicjalizacja samplera dla zrównoważenia klas
    print("Inicjalizacja samplera dla balansu klas...")
    class_to_weight = {cls: w for cls, w in zip(classes, weights)}
    sample_weights = [class_to_weight[label] for label in train_df['dx']]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Inicjalizacja DataLoaderów
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Gotowe! Załadowano {len(train_dataset)} próbek treningowych, "
          f"{len(val_dataset)} walidacyjnych, {len(test_dataset)} testowych.")

    return train_loader, val_loader, test_loader, class_weights, classes


if __name__ == "__main__":
    METADATA_PATH = r'HAM10000\HAM10000_metadata.csv'
    CLEAN_IMAGE_DIR = r'HAM10000\HAM10000_images_clean'
    MASK_DIR = r'HAM10000\HAM10000_segmentations_lesion_tschandl'

    t_loader, v_loader, test_loader, weights, cls_names = get_dataloaders_and_weights(
        metadata_path=METADATA_PATH,
        image_dir=CLEAN_IMAGE_DIR,
        mask_dir=MASK_DIR,
        batch_size=32
    )

    print(f"Wagi klas: {weights.tolist()}")
    print(f"Kolejność klas: {cls_names.tolist()}")