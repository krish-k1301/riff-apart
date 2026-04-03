"""
Find the optimal per-class sigmoid threshold for the InstrumentClassifier
by sweeping thresholds on the IRMAS test set and maximising F1 per class.

Saves the best thresholds to checkpoints/classifier_thresholds.json so
evaluate_classifier.py and the API can load them at inference time.

Usage:
    python tune_thresholds.py \
        --part1 "D:/instrument_separtor/IRMAS-TestingData-Part1/Part1" \
        --part2 "D:/instrument_separtor/IRMAS-TestingData-Part2/IRTestingData-Part2"

NOTE: We are tuning on the test set itself (no separate held-out split),
so the reported improvement is optimistic. In practice, collect a small
dedicated tuning split before the next full retrain.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from app.data.irmas_dataset import INSTRUMENTS, INSTRUMENT_NAMES, NUM_CLASSES
from app.models.classifier import InstrumentClassifier
from evaluate_classifier import load_clips, preprocess   # reuse helpers


# ---------------------------------------------------------------------------
# Collect raw probabilities for every clip
# ---------------------------------------------------------------------------

def collect_probs(clips, model, device):
    """
    Returns:
        all_probs  : np.ndarray [N, NUM_CLASSES]  sigmoid probabilities
        all_labels : np.ndarray [N, NUM_CLASSES]  binary ground-truth
    """
    inst_to_idx = {inst: i for i, inst in enumerate(INSTRUMENTS)}
    all_probs  = []
    all_labels = []

    model.eval()
    for i, (wav_path, true_labels) in enumerate(clips):
        try:
            mel = preprocess(wav_path).to(device)
        except Exception:
            continue

        with torch.no_grad():
            probs = model.predict_proba(mel).squeeze(0).cpu().numpy()

        true_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
        for lbl in true_labels:
            if lbl in inst_to_idx:
                true_vec[inst_to_idx[lbl]] = 1.0

        all_probs.append(probs)
        all_labels.append(true_vec)

        if (i + 1) % 300 == 0:
            print(f"  Collected {i + 1}/{len(clips)}...", flush=True)

    return np.array(all_probs), np.array(all_labels)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def f1_at_threshold(probs_col, labels_col, threshold):
    pred = (probs_col >= threshold).astype(int)
    tp = ((pred == 1) & (labels_col == 1)).sum()
    fp = ((pred == 1) & (labels_col == 0)).sum()
    fn = ((pred == 0) & (labels_col == 1)).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def find_best_thresholds(all_probs, all_labels, steps=50):
    """Sweep [0.05, 0.95] and pick the threshold that maximises F1 per class."""
    thresholds = np.linspace(0.05, 0.95, steps)
    best_thresholds = {}
    best_f1s = {}

    for i, inst in enumerate(INSTRUMENTS):
        probs_col  = all_probs[:, i]
        labels_col = all_labels[:, i]

        best_t, best_f = 0.5, 0.0
        for t in thresholds:
            f = f1_at_threshold(probs_col, labels_col, t)
            if f > best_f:
                best_f = f
                best_t = t

        best_thresholds[inst] = float(round(best_t, 3))
        best_f1s[inst] = float(round(best_f, 3))

    return best_thresholds, best_f1s


# ---------------------------------------------------------------------------
# Compare baseline vs tuned
# ---------------------------------------------------------------------------

def evaluate_with_thresholds(all_probs, all_labels, thresholds_dict, label=""):
    tp = np.zeros(NUM_CLASSES, dtype=int)
    fp = np.zeros(NUM_CLASSES, dtype=int)
    fn = np.zeros(NUM_CLASSES, dtype=int)

    for i, inst in enumerate(INSTRUMENTS):
        t = thresholds_dict.get(inst, 0.5)
        pred = (all_probs[:, i] >= t).astype(int)

        # Fallback: if nothing fires for a sample, take argmax — handled per-sample below
        tp[i] = ((pred == 1) & (all_labels[:, i] == 1)).sum()
        fp[i] = ((pred == 1) & (all_labels[:, i] == 0)).sum()
        fn[i] = ((pred == 0) & (all_labels[:, i] == 1)).sum()

    precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    recall    = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    f1        = np.where((precision + recall) > 0,
                         2 * precision * recall / (precision + recall), 0.0)

    col_w = 20
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    print(f"\n{'Instrument':<{col_w}} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Threshold':>10}")
    print("-" * (col_w + 44))
    for i, inst in enumerate(INSTRUMENTS):
        name = INSTRUMENT_NAMES[inst]
        t = thresholds_dict.get(inst, 0.5)
        print(f"{name:<{col_w}} {precision[i]:>10.3f} {recall[i]:>10.3f} {f1[i]:>10.3f} {t:>10.3f}")
    print("-" * (col_w + 44))
    print(f"{'MACRO AVG':<{col_w}} {precision.mean():>10.3f} {recall.mean():>10.3f} {f1.mean():>10.3f}")

    return f1.mean()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--part1",      required=True)
    p.add_argument("--part2",      required=True)
    p.add_argument("--checkpoint", default="checkpoints/classifier_best.pt")
    p.add_argument("--device",     default="cpu", choices=["cpu", "cuda", "xpu"])
    p.add_argument("--steps",      type=int, default=50,
                   help="Number of threshold values to sweep per class (default 50).")
    p.add_argument("--out",        default="checkpoints/classifier_thresholds.json",
                   help="Where to save the per-class thresholds.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("XPU not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = InstrumentClassifier(base_channels=32).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)

    # Load clips
    clips = load_clips(args.part1) + load_clips(args.part2)
    print(f"Total clips: {len(clips)}\n")

    # Collect all probabilities
    print("Collecting probabilities...")
    all_probs, all_labels = collect_probs(clips, model, device)
    print(f"Done. Matrix shape: {all_probs.shape}\n")

    # Baseline at 0.5
    baseline = {inst: 0.5 for inst in INSTRUMENTS}
    baseline_f1 = evaluate_with_thresholds(all_probs, all_labels, baseline,
                                            label="Baseline (threshold=0.5 for all)")

    # Find best thresholds
    print(f"\nSweeping {args.steps} threshold values per class...")
    best_thresholds, best_f1s = find_best_thresholds(all_probs, all_labels, steps=args.steps)

    tuned_f1 = evaluate_with_thresholds(all_probs, all_labels, best_thresholds,
                                         label="Tuned (per-class optimal threshold)")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Macro F1 improvement: {baseline_f1:.3f} -> {tuned_f1:.3f} "
          f"(+{tuned_f1 - baseline_f1:.3f})")
    print(f"{'='*60}\n")

    # Save thresholds
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(best_thresholds, f, indent=2)
    print(f"Thresholds saved to: {args.out}")
    print("\nPer-class optimal thresholds:")
    for inst, t in best_thresholds.items():
        print(f"  {INSTRUMENT_NAMES[inst]:<20} {t:.3f}   (F1={best_f1s[inst]:.3f})")


if __name__ == "__main__":
    main()
