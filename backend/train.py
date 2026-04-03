"""
Training script for the U-Net stem separator.

Train one model per target stem (vocals, drums, bass, other).

Example:
    python train.py --target vocals --data-root C:/Users/Krish Kubadia/MUSDB18/MUSDB18-7
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Ensure the backend/app package is importable when running from backend/
sys.path.insert(0, str(Path(__file__).parent))

from app.data.dataloader import get_dataloaders
from app.models.unet import UNet


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the U-Net stem separator.")

    parser.add_argument(
        "--target",
        required=True,
        choices=["vocals", "drums", "bass", "other"],
        help="Which stem to separate.",
    )
    parser.add_argument(
        "--data-root",
        default=r"C:\Users\Krish Kubadia\MUSDB18\MUSDB18-7",
        help="Path to the MUSDB18 dataset root.",
    )
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--batch-size",     type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--chunk-duration", type=float, default=5.0,
                        help="Audio chunk length in seconds.")
    parser.add_argument("--sr",             type=int,   default=44100)
    parser.add_argument("--n-fft",          type=int,   default=2048)
    parser.add_argument("--hop-length",     type=int,   default=512)
    parser.add_argument("--num-workers",    type=int,   default=0,
                        help="DataLoader workers (keep 0 on Windows).")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory for saving checkpoints.")
    parser.add_argument("--resume",         default=None,
                        help="Path to a checkpoint to resume training from.")
    parser.add_argument("--lr-patience",    type=int,   default=5,
                        help="ReduceLROnPlateau patience (epochs).")
    parser.add_argument("--device",         type=str,   default="xpu",
                        choices=["xpu", "cpu", "cuda"],
                        help="Device to train on (default: xpu).")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training / validation helpers
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0

    with torch.set_grad_enabled(train):
        for batch in loader:
            mix_mag    = batch["mix_mag"].to(device)     # [B, 1, F, T]
            target_mag = batch["target_mag"].to(device)  # [B, 1, F, T]

            mask = model(mix_mag)                        # [B, 1, F, T]
            pred = mask * mix_mag                        # estimated target magnitude

            loss = F.l1_loss(pred, target_mag)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, train_loss, val_loss, target):
    torch.save(
        {
            "epoch":                epoch,
            "target":               target,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss":           train_loss,
            "val_loss":             val_loss,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("XPU not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Target: {args.target}")

    # Dataloaders
    print("Loading dataset...")
    loaders = get_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        chunk_duration=args.chunk_duration,
        sr=args.sr,
        num_workers=args.num_workers,
        target=args.target,
    )
    train_loader = loaders["train"]
    val_loader   = loaders["test"]
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # Model, optimiser, scheduler
    model     = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )

    start_epoch = 1
    best_val_loss = float("inf")

    # Checkpoint directory
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path   = ckpt_dir / f"unet_{args.target}_best.pt"
    latest_path = ckpt_dir / f"unet_{args.target}_latest.pt"

    # Resume
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}, best val loss {best_val_loss:.6f}")

    end_epoch = max(args.epochs, start_epoch)
    print(f"\nTraining from epoch {start_epoch} to {end_epoch}...\n")
    print(f"{'Epoch':>6}  {'Train L1':>10}  {'Val L1':>10}  {'LR':>10}  {'Saved':>6}")
    print("-" * 52)

    for epoch in range(start_epoch, end_epoch + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss   = run_epoch(model, val_loader,   optimizer, device, train=False)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        saved_tag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, train_loss, val_loss, args.target)
            saved_tag = "best"

        save_checkpoint(latest_path, model, optimizer, scheduler, epoch, train_loss, val_loss, args.target)

        print(f"{epoch:>6}  {train_loss:>10.6f}  {val_loss:>10.6f}  {current_lr:>10.2e}  {saved_tag:>6}")

    print(f"\nDone. Best val L1: {best_val_loss:.6f}")
    print(f"Best checkpoint : {best_path}")


if __name__ == "__main__":
    main()
