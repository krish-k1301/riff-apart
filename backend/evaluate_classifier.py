"""
Evaluate the InstrumentClassifier against the IRMAS test set (Parts 1 & 2).

IRMAS test clips are multi-label — each .txt file lists one or more
instrument codes (one per line). This script reports:

  - Per-class Precision, Recall, F1 at threshold 0.5
  - Macro-averaged P / R / F1
  - Top-1 Recall  : true label set contains the top predicted instrument
  - Top-3 Recall  : true label set overlaps with top-3 predictions

Usage:
    python evaluate_classifier.py \
        --part1 D:/instrument_separtor/IRMAS-TestingData-Part1/Part1 \
        --part2 D:/instrument_separtor/IRMAS-TestingData-Part2/IRTestingData-Part2 \
        --checkpoint checkpoints/classifier_best.pt
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))

from app.data.irmas_dataset import IRMASDataset, INSTRUMENTS, INSTRUMENT_NAMES, NUM_CLASSES
from app.models.classifier import InstrumentClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_clips(folder: str):
    """
    Scan a flat IRMAS test folder and return list of (wav_path, label_set).
    label_set is a set of instrument codes, e.g. {"gel", "voi"}.
    """
    folder = Path(folder)
    clips = []
    for txt_path in sorted(folder.glob("*.txt")):
        wav_path = txt_path.with_suffix(".wav")
        if not wav_path.exists():
            continue
        labels = set()
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            code = line.strip()
            if code in INSTRUMENTS:
                labels.add(code)
        if labels:
            clips.append((str(wav_path), labels))
    return clips


def preprocess(wav_path: str, target_sr=22050, clip_samples=None) -> torch.Tensor:
    """
    Replicate IRMASDataset preprocessing: load → mono → resample → crop/pad → log-mel.
    Returns [1, 1, n_mels, T] ready for the model.
    """
    if clip_samples is None:
        clip_samples = int(3.0 * target_sr)

    audio, file_sr = sf.read(wav_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # stereo → mono

    if file_sr != target_sr:
        target_len = int(len(audio) * target_sr / file_sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    if len(audio) < clip_samples:
        audio = np.pad(audio, (0, clip_samples - len(audio)))
    else:
        audio = audio[:clip_samples]

    # Reuse IRMASDataset mel logic without scanning any folder
    dummy = IRMASDataset.__new__(IRMASDataset)
    dummy.sr = target_sr
    dummy.n_fft = 1024
    dummy.hop_length = 512
    dummy.n_mels = 128
    mel = dummy._mel_spectrogram(audio)  # [1, n_mels, T]
    return mel.unsqueeze(0)              # [1, 1, n_mels, T]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(clips, model, device, threshold=0.5, thresholds_dict=None):
    """
    thresholds_dict: optional {instrument_code: float} mapping loaded from
                     classifier_thresholds.json. Overrides the global threshold
                     on a per-class basis when provided.
    """
    model.eval()
    inst_to_idx = {inst: i for i, inst in enumerate(INSTRUMENTS)}

    # Accumulators per class: TP, FP, FN
    tp = np.zeros(NUM_CLASSES, dtype=int)
    fp = np.zeros(NUM_CLASSES, dtype=int)
    fn = np.zeros(NUM_CLASSES, dtype=int)

    top1_hits = 0
    top3_hits = 0
    total = 0
    errors = 0

    for i, (wav_path, true_labels) in enumerate(clips):
        try:
            mel = preprocess(wav_path).to(device)
        except Exception as e:
            errors += 1
            continue

        with torch.no_grad():
            probs = model.predict_proba(mel).squeeze(0).cpu().numpy()  # [11]

        # True label binary vector
        true_vec = np.zeros(NUM_CLASSES, dtype=int)
        for lbl in true_labels:
            if lbl in inst_to_idx:
                true_vec[inst_to_idx[lbl]] = 1

        # Predicted binary vector — per-class threshold if provided, else global
        if thresholds_dict:
            thresholds_vec = np.array([thresholds_dict.get(inst, threshold) for inst in INSTRUMENTS])
        else:
            thresholds_vec = np.full(NUM_CLASSES, threshold)
        pred_vec = (probs >= thresholds_vec).astype(int)

        # If nothing crosses threshold, take argmax as fallback
        if pred_vec.sum() == 0:
            pred_vec[probs.argmax()] = 1

        # Per-class TP/FP/FN
        tp += (pred_vec == 1) & (true_vec == 1)
        fp += (pred_vec == 1) & (true_vec == 0)
        fn += (pred_vec == 0) & (true_vec == 1)

        # Top-1 recall: highest-prob instrument in true set?
        top1_pred = INSTRUMENTS[probs.argmax()]
        if top1_pred in true_labels:
            top1_hits += 1

        # Top-3 recall: any of top-3 in true set?
        top3_preds = {INSTRUMENTS[j] for j in np.argsort(probs)[-3:]}
        if top3_preds & true_labels:
            top3_hits += 1

        total += 1

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(clips)} clips...", flush=True)

    return tp, fp, fn, top1_hits, top3_hits, total, errors


def print_results(tp, fp, fn, top1_hits, top3_hits, total, errors):
    precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    recall    = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    f1        = np.where((precision + recall) > 0,
                         2 * precision * recall / (precision + recall), 0.0)

    col_w = 20
    print()
    print(f"{'Instrument':<{col_w}} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * (col_w + 46))
    for i, inst in enumerate(INSTRUMENTS):
        name = INSTRUMENT_NAMES[inst]
        print(f"{name:<{col_w}} {precision[i]:>10.3f} {recall[i]:>10.3f} {f1[i]:>10.3f} {tp[i]:>6} {fp[i]:>6} {fn[i]:>6}")

    print("-" * (col_w + 46))
    print(f"{'MACRO AVG':<{col_w}} {precision.mean():>10.3f} {recall.mean():>10.3f} {f1.mean():>10.3f}")
    print()
    print(f"Clips evaluated : {total}  (skipped {errors} errors)")
    print(f"Top-1 Recall    : {top1_hits}/{total} = {top1_hits/total*100:.1f}%")
    print(f"Top-3 Recall    : {top3_hits}/{total} = {top3_hits/total*100:.1f}%")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--part1",       required=True, help="Path to IRMAS test Part1 flat folder")
    p.add_argument("--part2",       required=True, help="Path to IRMAS test Part2 flat folder")
    p.add_argument("--checkpoint",  default="checkpoints/classifier_best.pt")
    p.add_argument("--threshold",   type=float, default=0.5,
                   help="Sigmoid threshold for positive prediction (default 0.5)")
    p.add_argument("--thresholds",  default=None,
                   help="Path to per-class thresholds JSON (from tune_thresholds.py). "
                        "Overrides --threshold when provided.")
    p.add_argument("--device",      default="cpu", choices=["cpu", "cuda", "xpu"])
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
    print("Model loaded.\n")

    # Load clips from both parts
    print(f"Scanning Part1: {args.part1}")
    clips1 = load_clips(args.part1)
    print(f"  {len(clips1)} clips found")

    print(f"Scanning Part2: {args.part2}")
    clips2 = load_clips(args.part2)
    print(f"  {len(clips2)} clips found")

    clips = clips1 + clips2
    print(f"\nTotal clips: {len(clips)}")
    print(f"Device     : {device}\n")

    # Load per-class thresholds if provided
    thresholds_dict = None
    if args.thresholds:
        import json
        with open(args.thresholds) as f:
            thresholds_dict = json.load(f)
        print(f"Using per-class thresholds from: {args.thresholds}")
    else:
        print(f"Threshold  : {args.threshold}")

    print("Evaluating...")
    tp, fp, fn, top1, top3, total, errors = evaluate(
        clips, model, device, args.threshold, thresholds_dict
    )
    print_results(tp, fp, fn, top1, top3, total, errors)


if __name__ == "__main__":
    main()
