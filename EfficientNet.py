import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import models
from Preprocessing import get_dataloaders_and_weights, set_seed

set_seed(42)

# Focal Loss, wasn't used at the end
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_efficientnet_model(num_classes=7):
    print("Downloading pre-trained model EfficientNet-B3...")
    model = models.efficientnet_b3(weights='DEFAULT')

    for param in model.features[7].parameters():
        param.requires_grad = True

    for param in model.features[8].parameters():
        param.requires_grad = True

    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )

    return model


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=100, patience=10):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    epochs_no_improve = 0

    scaler = torch.amp.GradScaler('cuda')

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f'Epoch {epoch + 1}/{num_epochs} | LR: {current_lr:.6f}')

        val_epoch_loss = 0.0

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            dataloader = train_loader if phase == 'train' else val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        #loss.backward()
                        #optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = (running_corrects.double() / len(dataloader.dataset)).item()

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
                print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            else:
                val_epoch_loss = epoch_loss
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                print(f'Val   Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_epoch_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        if new_lr < old_lr:
            print(f'-> Learning rate decreased: {old_lr:.6f} -> {new_lr:.6f}')

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f'-> No improvement for {epochs_no_improve} epoch(s).')

        if epochs_no_improve >= patience:
            print(f'\nEarly stopping at epoch {epoch + 1}!')
            break
        print('-' * 30)

    print(f'Best Validation Loss achieved: {best_val_loss:.4f}')
    model.load_state_dict(best_model_wts)
    return model, history


# FUNKCJE WIZUALIZACJI I EWALUACJI
def plot_training_history(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (EfficientNet)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy (EfficientNet)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    chart_path = os.path.join(save_dir, 'EfficientNet_Acc.png')
    plt.savefig(chart_path)
    plt.close()
    print(f"-> Saved training history plot as {chart_path}")


def evaluate_and_save_model(model, test_loader, device, classes, save_dir):
    print("\nEvaluating the model on the test set...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=classes, zero_division=0)
    print("\n   CLASSIFICATION REPORT     ")
    print(report)

    report_path = os.path.join(save_dir, 'EfficientNet_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("   Classification Report (EfficientNet)   \n")
        f.write(report)
    print(f"-> Saved classification report: {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (EfficientNet)')
    plt.xlabel('Predicted Classification')
    plt.ylabel('Actual Classification')
    plt.tight_layout()

    cm_path = os.path.join(save_dir, 'EfficientNet_CM.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"-> Saved confusion matrix: {cm_path}")


# GŁÓWNY PROCES URUCHOMIENIA
if __name__ == "__main__":
    METADATA_PATH = r'HAM10000\HAM10000_metadata.csv'
    CLEAN_IMAGE_DIR = r'HAM10000\HAM10000_images_mask'
    #MASK_DIR = r'HAM10000\HAM10000_segmentations_lesion_tschandl'
    MODEL_DIR = r'Models\EfficientNet\Final'

    os.makedirs(MODEL_DIR, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    print("\nPreparing data...")

    train_loader, val_loader, test_loader, class_weights, classes = get_dataloaders_and_weights(
        metadata_path=METADATA_PATH,
        image_dir=CLEAN_IMAGE_DIR,
        #mask_dir=MASK_DIR,
        img_size=260,
        seg_mode='none',
        use_sampler=True
    )

    class_weights = class_weights.to(device)
    num_classes = len(classes)

    model = get_efficientnet_model(num_classes=num_classes).to(device)

    #criterion = FocalLoss(alpha=None, gamma=2.0)
    #criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.00001,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    print("\nStarting training...")
    trained_model, training_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=1000,
        patience=10
    )

    # Zapis historii do JSON
    history_path = os.path.join(MODEL_DIR, 'EfficientNet_report.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=4)
    print(f"\n-> Saved raw training data to: {history_path}")

    # Rysowanie wykresów i ewaluacja z zapisem do txt
    plot_training_history(training_history, MODEL_DIR)
    evaluate_and_save_model(trained_model, test_loader, device, classes, MODEL_DIR)

    # Zapis modelu
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'EfficientNet_model.pth')
    torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nBest model saved as: {MODEL_SAVE_PATH}")