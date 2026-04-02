"""
Train the InstrumentClassifier on IRMAS training data.

Usage:
    python train_classifier.py --data-root D:/instrument_separtor/IRMAS-TrainingData
    python train_classifier.py --data-root D:/instrument_separtor/IRMAS-TrainingData --epochs 30 --resume
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(__file__))

from app.data.irmas_dataset import IRMASDataset, INSTRUMENTS, INSTRUMENT_NAMES
from app.models.classifier import InstrumentClassifier

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="Path to IRMAS-TrainingData folder")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.1, help="Fraction of data for validation")
    p.add_argument("--resume", action="store_true", help="Resume from classifier_latest.pt")
    p.add_argument("--device", type=str, default="xpu",
                   choices=["xpu", "cpu", "cuda"],
                   help="Device to train on (default: xpu)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("XPU not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Instruments ({len(INSTRUMENTS)}): {[INSTRUMENT_NAMES[i] for i in INSTRUMENTS]}")

    # Dataset
    full_dataset = IRMASDataset(root=args.data_root)
    print(f"Total clips: {len(full_dataset)}")

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    # Model
    model = InstrumentClassifier(base_channels=32).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    start_epoch = 0
    best_val_loss = float("inf")

    latest_path = os.path.join(CHECKPOINT_DIR, "classifier_latest.pt")
    if args.resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for mel, label in train_loader:
            mel, label = mel.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(mel)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * mel.size(0)
        train_loss /= len(train_ds)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for mel, label in val_loader:
                mel, label = mel.to(device), label.to(device)
                logits = model(mel)
                loss = criterion(logits, label)
                val_loss += loss.item() * mel.size(0)

                # Accuracy: predicted class == true class (single-label IRMAS training)
                preds = logits.argmax(dim=1)
                targets = label.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += mel.size(0)

        val_loss /= len(val_ds)
        accuracy = correct / total * 100
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {accuracy:.1f}%"
        )

        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }, latest_path)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "classifier_best.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, best_path)
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")

    print("Training complete.")


if __name__ == "__main__":
    main()
