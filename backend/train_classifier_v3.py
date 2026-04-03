"""
Train the InstrumentClassifier v3 — same as v2 (SpecAugment + Mixup) but
with base_channels=64 for greater model capacity (~5.7M params vs 1.44M).

Saves to classifier_v3_best.pt / classifier_v3_latest.pt.

Usage:
    python train_classifier_v3.py --data-root D:/instrument_separtor/IRMAS-TrainingData
    python train_classifier_v3.py --data-root D:/instrument_separtor/IRMAS-TrainingData --resume
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(__file__))

from app.data.irmas_dataset import IRMASDataset, INSTRUMENTS, INSTRUMENT_NAMES
from app.models.classifier import InstrumentClassifier
from train_classifier_v2 import spec_augment, mixup_batch  # reuse augmentation helpers

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Train InstrumentClassifier v3 (base_channels=64)")
    p.add_argument("--data-root",   required=True)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--val-split",   type=float, default=0.1)
    p.add_argument("--freq-mask",   type=int,   default=20)
    p.add_argument("--time-mask",   type=int,   default=30)
    p.add_argument("--mixup-alpha", type=float, default=0.4)
    p.add_argument("--resume",      action="store_true",
                   help="Resume from classifier_v3_latest.pt")
    p.add_argument("--device",      type=str,   default="xpu",
                   choices=["xpu", "cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("XPU not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device       : {device}")
    print(f"base_channels: 64  (~5.7M params)")
    print(f"SpecAugment  : freq_mask={args.freq_mask}  time_mask={args.time_mask}")
    print(f"Mixup alpha  : {args.mixup_alpha}\n")

    # Dataset — same 90/10 split seed as v1/v2 for fair comparison
    full_dataset = IRMASDataset(root=args.data_root)
    print(f"Total clips : {len(full_dataset)}")

    val_size   = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Larger model
    model     = InstrumentClassifier(base_channels=64).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters  : {total_params:,}\n")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=4, factor=0.5
    )

    start_epoch   = 0
    best_val_loss = float("inf")

    latest_path = os.path.join(CHECKPOINT_DIR, "classifier_v3_latest.pt")
    best_path   = os.path.join(CHECKPOINT_DIR, "classifier_v3_best.pt")

    if args.resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, best val loss {best_val_loss:.4f}\n")

    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>9}  {'Val Acc':>8}  {'LR':>8}  {'Saved':>5}")
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs):

        # ── Train ────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for mel, label in train_loader:
            mel, label = mel.to(device), label.to(device)
            mel         = spec_augment(mel, args.freq_mask, args.time_mask)
            if args.mixup_alpha > 0:
                mel, label = mixup_batch(mel, label, args.mixup_alpha)

            optimizer.zero_grad()
            loss = criterion(model(mel), label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * mel.size(0)

        train_loss /= len(train_ds)

        # ── Validate ─────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0

        with torch.no_grad():
            for mel, label in val_loader:
                mel, label = mel.to(device), label.to(device)
                logits     = model(mel)
                val_loss  += criterion(logits, label).item() * mel.size(0)

                preds   = logits.argmax(dim=1)
                targets = label.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total   += mel.size(0)

        val_loss  /= len(val_ds)
        accuracy   = correct / total * 100
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        saved_tag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, best_path)
            saved_tag = "best"

        torch.save({
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss":   best_val_loss,
        }, latest_path)

        print(
            f"{epoch+1:>6}  {train_loss:>11.4f}  {val_loss:>9.4f}  "
            f"{accuracy:>7.1f}%  {current_lr:>8.2e}  {saved_tag:>5}"
        )

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint : {best_path}")


if __name__ == "__main__":
    main()
