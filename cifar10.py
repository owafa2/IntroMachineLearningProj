"""
CIFAR-10 Classification Project
Two models compared: a small CNN trained from scratch vs. a fine-tuned ResNet-18.

Outputs (saved to ./outputs/):
  - metrics.json           : accuracy, macro P/R/F1, 95% bootstrap CIs for both models
  - confmat_<model>.png    : normalized confusion matrix per model
  - curves_<model>.png     : train/val loss & accuracy curves per model
  - <model>_best.pt        : best checkpoint per model

Run:
  python cifar10_project.py --epochs_cnn 30 --epochs_resnet 15
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

# ----------------------------- setup -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


# ----------------------------- data -----------------------------
def get_loaders(batch_size=128, resnet_norm=False):
    """resnet_norm=True uses ImageNet stats + 224x224 resize for the pretrained model."""
    if resnet_norm:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        train_tf = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=8),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    train_full = datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
    test_set   = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)

    # 45k train / 5k val split
    n_val = 5000
    n_train = len(train_full) - n_val
    train_set, val_set = random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    # val should use test transform; quick swap
    val_set.dataset = datasets.CIFAR10("./data", train=True, download=False, transform=test_tf)

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True),
        DataLoader(val_set,   batch_size=256,        shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(test_set,  batch_size=256,        shuffle=False, num_workers=2, pin_memory=True),
    )


# ----------------------------- models -----------------------------
class SmallCNN(nn.Module):
    """3 conv blocks + classifier. ~500k params."""
    def __init__(self, num_classes=10):
        super().__init__()
        def block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(3, 64),     # 32 -> 16
            block(64, 128),   # 16 -> 8
            block(128, 256),  # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))


def build_resnet18():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m


# ----------------------------- train / eval -----------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def train_model(name, model, train_loader, val_loader, epochs, lr):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_path = 0.0, OUT / f"{name}_best.pt"

    for ep in range(1, epochs + 1):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer)
        vl, va = run_epoch(model, val_loader,   criterion)
        scheduler.step()
        history["train_loss"].append(tl); history["train_acc"].append(ta)
        history["val_loss"].append(vl);   history["val_acc"].append(va)
        print(f"[{name}] epoch {ep:02d}/{epochs}  "
              f"train {tl:.3f}/{ta:.3f}  val {vl:.3f}/{va:.3f}")
        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    return model, history


@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        ps.append(model(x).argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


# ----------------------------- metrics -----------------------------
def bootstrap_ci(y_true, y_pred, n_boot=1000, alpha=0.05, seed=SEED):
    """95% bootstrap CI for accuracy by resampling the test set with replacement."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        accs[i] = (y_true[idx] == y_pred[idx]).mean()
    lo = np.quantile(accs, alpha / 2)
    hi = np.quantile(accs, 1 - alpha / 2)
    return float(lo), float(hi)


def plot_confmat(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Normalized Confusion Matrix — {name}")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / f"confmat_{name}.png", dpi=150)
    plt.close(fig)


def plot_curves(history, name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title(f"{name} loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"],   label="val")
    axes[1].set_title(f"{name} accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / f"curves_{name}.png", dpi=150)
    plt.close(fig)


def evaluate(name, model, test_loader):
    y_true, y_pred = predict(model, test_loader)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    lo, hi = bootstrap_ci(y_true, y_pred)
    plot_confmat(y_true, y_pred, name)
    return {
        "accuracy": float(acc),
        "accuracy_ci95": [lo, hi],
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
    }


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs_cnn",    type=int, default=30)
    ap.add_argument("--epochs_resnet", type=int, default=15)
    ap.add_argument("--batch_size",    type=int, default=128)
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    results = {}

    # ---- Model 1: SmallCNN from scratch (32x32) ----
    tr, va, te = get_loaders(args.batch_size, resnet_norm=False)
    cnn, hist_cnn = train_model("smallcnn", SmallCNN(),
                                tr, va, args.epochs_cnn, lr=0.05)
    plot_curves(hist_cnn, "smallcnn")
    results["smallcnn"] = evaluate("smallcnn", cnn, te)

    # ---- Model 2: ResNet-18 fine-tuned (224x224, ImageNet pretrained) ----
    tr, va, te = get_loaders(args.batch_size, resnet_norm=True)
    rn, hist_rn = train_model("resnet18", build_resnet18(),
                              tr, va, args.epochs_resnet, lr=0.01)
    plot_curves(hist_rn, "resnet18")
    results["resnet18"] = evaluate("resnet18", rn, te)

    with open(OUT / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== FINAL ===")
    for name, m in results.items():
        lo, hi = m["accuracy_ci95"]
        print(f"{name:10s}  acc={m['accuracy']:.4f}  "
              f"95% CI=[{lo:.4f},{hi:.4f}]  F1={m['f1_macro']:.4f}")


if __name__ == "__main__":
    main()