import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# CONFIG  <- change DATASET_DIR to your path
# ──────────────────────────────────────────────
DATASET_DIR = r"C:\Users\PRANAV\Downloads\cone_depression\cone_depression\data_main"
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-3
IMG_SIZE    = 128
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH   = "model.pth"
CURVES_PATH = "training_curves.png"

# ──────────────────────────────────────────────
# MODEL  (defined outside main so evaluate.py can import it)
# ──────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ──────────────────────────────────────────────
# MAIN GUARD  (required on Windows)
# ──────────────────────────────────────────────
if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
    num_classes  = len(full_dataset.classes)

    train_size = int(0.7 * len(full_dataset))
    test_size  = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # num_workers=0 fixes the Windows multiprocessing crash
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Classes : {full_dataset.classes}")
    print(f"Train   : {train_size}  |  Test: {test_size}")
    print(f"Device  : {DEVICE}")

    model     = SimpleCNN(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history  = {"train_loss": [], "train_acc": [], "test_acc": []}
    best_acc = 0.0

    for epoch in range(EPOCHS):

        # ── train ──
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted  = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc  = correct / total

        # ── eval ──
        model.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct_test += (predicted == labels).sum().item()
                total_test   += labels.size(0)

        test_acc = correct_test / total_test

        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch+1:02d}/{EPOCHS}  "
              f"Loss: {epoch_loss:.4f}  "
              f"TrainAcc: {epoch_acc:.4f}  "
              f"TestAcc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> Best model saved  (TestAcc={best_acc:.4f})")

    # ── training curves ──
    epochs_range = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_range, history["train_loss"], marker="o", label="Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs_range, history["train_acc"], marker="o", label="Train Acc")
    axes[1].plot(epochs_range, history["test_acc"],  marker="s", label="Test Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(CURVES_PATH, dpi=150)
    print(f"\nTraining curves saved -> {CURVES_PATH}")
    plt.show()