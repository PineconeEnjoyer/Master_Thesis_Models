import os
import numpy as np
import pandas as pd
import random
import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()


def get_transforms(img_size):
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=90, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_test_transforms

# This segmentation function was tested. However, results weren't satisfying enough to keep in the final thesis
def apply_segmentation(image_pil, mask_pil, mode='mask', margin=15):
    if image_pil.size != mask_pil.size:
        try:
            mask_pil = mask_pil.resize(image_pil.size, Image.Resampling.NEAREST)
        except AttributeError:
            mask_pil = mask_pil.resize(image_pil.size, Image.NEAREST)

    img = np.array(image_pil)
    mask = np.array(mask_pil.convert('L'))

    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    if mode == 'mask':
        segmented_img = cv2.bitwise_and(img, img, mask=mask_binary)
        return Image.fromarray(segmented_img)

    elif mode == 'crop':
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image_pil

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        h_img, w_img, _ = img.shape
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(w_img, x + w + margin)
        y_end = min(h_img, y + h + margin)

        cropped_img = img[y_start:y_end, x_start:x_end]
        return Image.fromarray(cropped_img)

    return image_pil


class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, image_dir, mask_dir=None, transform=None, classes=None, seg_mode='none',
                 img_size=224):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.seg_mode = seg_mode
        self.img_size = img_size

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

        try:
            image = Image.open(img_path).convert('RGB')

            if self.mask_dir and self.seg_mode != 'none':
                mask_path = os.path.join(self.mask_dir, f"{image_id}_segmentation.png")
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert('L')
                    image = apply_segmentation(image, mask, mode=self.seg_mode)
                else:
                    print(f"Error: No mask for {image_id}. Processing the original.")

        except FileNotFoundError:
            print(f"Error: No image file {img_path}")
            image = Image.new('RGB', (self.img_size, self.img_size))

        label_name = self.dataframe.iloc[idx]['dx']
        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_stratified_group_splits(metadata_path, test_size=0.15, val_size=0.15):
    df = pd.read_csv(metadata_path)
    lesion_df = df.groupby('lesion_id')['dx'].first().reset_index()
    temp_size = test_size + val_size

    train_lesions, temp_lesions = train_test_split(
        lesion_df, test_size=temp_size, stratify=lesion_df['dx'], random_state=42
    )
    val_ratio = val_size / temp_size
    val_lesions, test_lesions = train_test_split(
        temp_lesions, test_size=(1 - val_ratio), stratify=temp_lesions['dx'], random_state=42
    )

    train_df = df[df['lesion_id'].isin(train_lesions['lesion_id'])].reset_index(drop=True)
    val_df = df[df['lesion_id'].isin(val_lesions['lesion_id'])].reset_index(drop=True)
    test_df = df[df['lesion_id'].isin(test_lesions['lesion_id'])].reset_index(drop=True)

    print("Train Unique:", train_df.groupby('dx')['lesion_id'].nunique())
    print("Val Unique:", val_df.groupby('dx')['lesion_id'].nunique())
    print("Test Unique:", test_df.groupby('dx')['lesion_id'].nunique())

    return train_df, val_df, test_df


def get_dataloaders_and_weights(metadata_path, image_dir, mask_dir=None,
                                                                                                                                                                                                                                                                                                                                                                                                                        batch_size=32, num_workers=2,
                                img_size=224, seg_mode='none', use_sampler=False):

    print("Loading and split of metadata...")
    train_df, val_df, test_df = create_stratified_group_splits(metadata_path, test_size=0.15, val_size=0.15)

    classes = np.array(sorted(train_df['dx'].unique()))
    y_train = train_df['dx'].values

    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float)

    train_transforms, val_test_transforms = get_transforms(img_size)

    print(f"Dataset initialization (Size: {img_size}x{img_size}, Segmentation: {seg_mode})...")
    train_dataset = SkinLesionDataset(train_df, image_dir, mask_dir, transform=train_transforms, classes=classes,
                                      seg_mode=seg_mode, img_size=img_size)
    val_dataset = SkinLesionDataset(val_df, image_dir, mask_dir, transform=val_test_transforms, classes=classes,
                                    seg_mode=seg_mode, img_size=img_size)
    test_dataset = SkinLesionDataset(test_df, image_dir, mask_dir, transform=val_test_transforms, classes=classes,
                                     seg_mode=seg_mode, img_size=img_size)

    if use_sampler:
        print("Balance sampling enabled (WeightedRandomSampler).")
        class_to_weight = {cls: w for cls, w in zip(classes, weights)}
        sample_weights = [class_to_weight[label] for label in train_df['dx']]

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(
        f"Loaded {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test samples.")

    return train_loader, val_loader, test_loader, class_weights, classes