import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from Preprocessing import get_dataloaders_and_weights


# ARCHITEKTURA SIECI CNN
class ClassicSkinCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(ClassicSkinCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((7, 7))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# FUNKCJA TRENINGOWA
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=100, patience=10):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    epochs_no_improve = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f'Epoka {epoch + 1}/{num_epochs} | LR: {current_lr:.6f}')

        val_epoch_loss = 0.0

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

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

        # Scheduler: zmniejsza LR, jeśli val_loss przestaje się poprawiać
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_epoch_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        if new_lr < old_lr:
            print(f'-> Learning rate zmniejszony: {old_lr:.6f} -> {new_lr:.6f}')

        # Early Stopping i Model Checkpointing
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f'-> Brak poprawy od {epochs_no_improve} epok.')

        if epochs_no_improve >= patience:
            print(f'\n[!] Wczesne zatrzymanie w epoce {epoch + 1}!')
            break

        print('-' * 30)

    print(f'Najlepszy wynik Validation Loss: {best_val_loss:.4f}')
    model.load_state_dict(best_model_wts)
    return model, history


# FUNKCJE WIZUALIZACJI I EWALUACJI
def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (CNN)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy (CNN)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(r'Models\CNN\CNN_Acc.png')
    plt.close()

    print("-> Zapisano wykres historii uczenia jako 'CNN_Acc.png'")


def evaluate_model(model, test_loader, device, classes):
    print("\nTrwa ewaluacja modelu na zbiorze testowym...")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generowanie i zapisywanie raportu klasyfikacji
    report = classification_report(all_labels, all_preds, target_names=classes, zero_division=0)
    print("\n--- RAPORT KLASYFIKACJI ---")
    print(report)

    report_path = os.path.join(r'Models\CNN\CNN_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("--- Classification Report (CNN) ---\n")
        f.write(report)
    print(f"-> Zapisano raport klasyfikacji: {report_path}")

    # Generowanie i zapisywanie macierzy pomyłek
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Classification')
    plt.ylabel('Actual Classification')
    plt.tight_layout()
    plt.savefig(r'Models\CNN\CNN_CM.png')
    plt.close()

    print("-> Zapisano macierz pomyłek jako 'CNN_CM.png'")


# GŁÓWNY PROCES URUCHOMIENIA
if __name__ == "__main__":
    METADATA_PATH = r'HAM10000\HAM10000_metadata.csv'
    CLEAN_IMAGE_DIR = r'HAM10000\HAM10000_images_clean'
    MODEL_DIR = r'Models\CNN'

    os.makedirs(MODEL_DIR, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Uruchomiono na urządzeniu: {device}")

    print("\nTrwa przygotowywanie danych...")

    train_loader, val_loader, test_loader, class_weights, classes = get_dataloaders_and_weights(
        metadata_path=METADATA_PATH,
        image_dir=CLEAN_IMAGE_DIR,
        batch_size=32,
        num_workers=2
    )

    class_weights = class_weights.to(device)
    num_classes = len(classes)

    model = ClassicSkinCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optymalizator z regularyzacją L2
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0001,
        weight_decay=1e-4
    )

    # Poprawka:
    # Usunięto verbose=True, ponieważ Twoja wersja PyTorch go nie obsługuje.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=4
    )

    print("\nStart treningu...")

    trained_model, training_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=100,
        patience=10
    )

    # Zapis historii do JSON
    history_path = os.path.join(MODEL_DIR, 'CNN_report.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=4)
    print(f"\n-> Zapisano surowe dane uczenia do: {history_path}")

    # Rysowanie wykresów i ewaluacja z zapisem do txt
    plot_training_history(training_history)
    evaluate_model(trained_model, test_loader, device, classes)

    # Zapis modelu
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'CNN_model.pth')
    torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nSukces! Najlepszy model został zapisany pod nazwą: {MODEL_SAVE_PATH}")