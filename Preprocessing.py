import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader


# PODZIAŁ DANYCH (ZAPOBIEGANIE WYCIEKOM)

def create_stratified_group_splits(metadata_path, test_size=0.15, val_size=0.15):
    df = pd.read_csv(metadata_path)

    lesion_df = df.groupby('lesion_id')['dx'].first().reset_index()
    temp_size = test_size + val_size

    train_lesions, temp_lesions = train_test_split(
        lesion_df, test_size=temp_size, stratify=lesion_df['dx'], random_state=42
    )

    val_ratio = val_size / temp_size
    val_lesions, test_lesions = train_test_split(
        temp_lesions, test_size=(1.0 - val_ratio), stratify=temp_lesions['dx'], random_state=42
    )

    train_df = df[df['lesion_id'].isin(train_lesions['lesion_id'])].reset_index(drop=True)
    val_df = df[df['lesion_id'].isin(val_lesions['lesion_id'])].reset_index(drop=True)
    test_df = df[df['lesion_id'].isin(test_lesions['lesion_id'])].reset_index(drop=True)

    return train_df, val_df, test_df


# TRANSFORMACJE (TYLKO AUGMENTACJA I NORMALIZACJA)

IMG_SIZE = 224

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# DATASET W PYTORCH

class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, classes=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

        if classes is None:
            self.classes = sorted(self.dataframe['dx'].unique())
        else:
            self.classes = list(classes)

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = f"{self.dataframe.iloc[idx]['image_id']}.jpg"
        img_path = os.path.join(self.image_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Ostrzeżenie: Brak pliku {img_path}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))

        label_name = self.dataframe.iloc[idx]['dx']
        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label


# GŁÓWNY BLOK (URUCHOMIENIE)

if __name__ == "__main__":
    METADATA_PATH = 'HAM10000_metadata.csv'
    CLEAN_IMAGE_DIR = 'HAM10000_images_clean'

    try:
        train_df, val_df, test_df = create_stratified_group_splits(METADATA_PATH)

        classes = np.array(sorted(train_df['dx'].unique()))
        y_train = train_df['dx'].values

        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        class_weights = torch.tensor(weights, dtype=torch.float)

        train_dataset = SkinLesionDataset(train_df, CLEAN_IMAGE_DIR, transform=train_transforms, classes=classes)
        val_dataset = SkinLesionDataset(val_df, CLEAN_IMAGE_DIR, transform=val_test_transforms, classes=classes)
        test_dataset = SkinLesionDataset(test_df, CLEAN_IMAGE_DIR, transform=val_test_transforms, classes=classes)

        BATCH_SIZE = 32
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        print(
            f"Gotowe! Załadowano {len(train_dataset)} próbek treningowych, {len(val_dataset)} walidacyjnych, {len(test_dataset)} testowych."
        )

    except FileNotFoundError as e:
        print(f"Błąd: {e}")