"""
evaluate.py  -  Final metrics & failure analysis for SimpleCNN
Usage:
    python evaluate.py --model model.pth --data "C:\\path\\to\\data_main"
"""

import argparse, math, itertools
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support,
    roc_auc_score, roc_curve
)

# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="model.pth")
parser.add_argument("--data",  default=r"C:\Users\PRANAV\Downloads\cone_depression\cone_depression\data_main")
parser.add_argument("--batch", default=32, type=int)
parser.add_argument("--img",   default=128, type=int)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# MODEL  (must match train.py exactly)
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


if __name__ == '__main__':

    # ──────────────────────────────────────────────
    # DATA  (same seed -> same test split as training)
    # ──────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((args.img, args.img)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(args.data, transform=transform)
    num_classes  = len(full_dataset.classes)
    class_names  = full_dataset.classes

    train_size = int(0.7 * len(full_dataset))
    test_size  = len(full_dataset) - train_size
    _, test_dataset = random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # num_workers=0 fixes Windows crash
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    # ──────────────────────────────────────────────
    # LOAD MODEL
    # ──────────────────────────────────────────────
    model = SimpleCNN(num_classes).to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model.eval()
    print(f"Loaded: {args.model}  |  Classes: {class_names}")

    # ──────────────────────────────────────────────
    # INFERENCE
    # ──────────────────────────────────────────────
    inv_norm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std =[1/0.229,      1/0.224,      1/0.225]
    )

    all_labels, all_preds, all_probs, all_images_raw = [], [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            for img in images.cpu():
                all_images_raw.append(
                    inv_norm(img).clamp(0, 1).permute(1, 2, 0).numpy()
                )

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    # ──────────────────────────────────────────────
    # 1. CLASSIFICATION REPORT
    # ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )

    # ──────────────────────────────────────────────
    # 2. CONFUSION MATRIX
    # ──────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(max(6, num_classes), max(5, num_classes - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(class_names)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved -> confusion_matrix.png")

    # ──────────────────────────────────────────────
    # 3. PRECISION / RECALL / F1  per class
    # ──────────────────────────────────────────────
    x = np.arange(num_classes)
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, num_classes * 1.5), 5))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x,         recall,    width, label="Recall")
    ax.bar(x + width, f1,        width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("per_class_metrics.png", dpi=150)
    print("Saved -> per_class_metrics.png")

    # ──────────────────────────────────────────────
    # 4. AUC-ROC  (one-vs-rest per class)
    # ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    auc_scores = []
    for i, cls in enumerate(class_names):
        y_true_bin = (all_labels == i).astype(int)
        y_score    = all_probs[:, i]
        if y_true_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        auc = roc_auc_score(y_true_bin, y_score)
        auc_scores.append(auc)
        ax.plot(fpr, tpr, label=f"{cls}  (AUC={auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves  (mean AUC={np.mean(auc_scores):.3f})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150)
    print("Saved -> roc_curves.png")

    # ──────────────────────────────────────────────
    # 5. FAILURE EXAMPLES  (most confident mistakes)
    # ──────────────────────────────────────────────
    wrong_idx  = np.where(all_preds != all_labels)[0]
    wrong_conf = all_probs[wrong_idx, all_preds[wrong_idx]]
    order      = np.argsort(-wrong_conf)
    worst      = wrong_idx[order[:min(12, len(order))]]

    if len(worst) > 0:
        cols = 4
        rows = math.ceil(len(worst) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = np.array(axes).flatten()

        for ax_i, idx in enumerate(worst):
            axes[ax_i].imshow(all_images_raw[idx])
            axes[ax_i].set_title(
                f"True : {class_names[all_labels[idx]]}\n"
                f"Pred : {class_names[all_preds[idx]]}\n"
                f"Conf : {all_probs[idx, all_preds[idx]]:.2f}",
                fontsize=8
            )
            axes[ax_i].axis("off")

        for ax_i in range(len(worst), len(axes)):
            axes[ax_i].axis("off")

        plt.suptitle("Most Confident Mistakes", fontsize=12)
        plt.tight_layout()
        plt.savefig("failure_examples.png", dpi=150)
        print("Saved -> failure_examples.png")
    else:
        print("No mistakes found on test set!")

    # ──────────────────────────────────────────────
    # 6. SHORTCOMINGS SUMMARY
    # ──────────────────────────────────────────────
    overall_acc   = (all_preds == all_labels).mean()
    worst_f1_idx  = np.argmin(f1)

    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    max_i, max_j = np.unravel_index(cm_no_diag.argmax(), cm_no_diag.shape)

    print("\n" + "="*60)
    print("MODEL SHORTCOMINGS ANALYSIS")
    print("="*60)
    print(f"Overall Test Accuracy : {overall_acc:.4f}")
    print(f"Mean AUC-ROC          : {np.mean(auc_scores):.4f}")
    print(f"Weakest class (F1)    : '{class_names[worst_f1_idx]}'  (F1={f1[worst_f1_idx]:.3f})")
    print(f"Most confused pair    : '{class_names[max_i]}' -> '{class_names[max_j]}'  ({cm_no_diag[max_i,max_j]} times)")
   
    plt.show()
    print("Done. Check the saved .png files in your working directory.")