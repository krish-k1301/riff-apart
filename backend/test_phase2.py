"""
Phase 2 test script -- Data Pipeline
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.data.dataset import MUSDBDataset
from app.data.dataloader import get_dataloaders
from app.audio.utils import plot_spectrogram
from app.audio.processor import AudioProcessor

OUTPUTS = Path(__file__).parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

errors = []

# ── 1. Init dataset ───────────────────────────────────────────────────────
print("[1] Loading MUSDBDataset (download=True, split=train, target=vocals)...")
try:
    ds = MUSDBDataset(download=True, split="train", target="vocals", chunk_duration=5.0)
    n_tracks = len(ds)
    print(f"    Dataset length : {n_tracks} tracks")
    assert n_tracks > 0, "Dataset is empty"
except Exception as e:
    errors.append(f"Dataset init: {e}")
    print(f"    ERROR: {e}")
    ds = None

# ── 2. Sample at index 0 ──────────────────────────────────────────────────
sample = None
if ds is not None:
    print("\n[2] Fetching sample at index 0...")
    try:
        sample = ds[0]
        mm  = sample["mix_mag"]
        tm  = sample["target_mag"]
        mp  = sample["mix_phase"]
        print(f"    mix_mag    shape : {list(mm.shape)}  dtype={mm.dtype}")
        print(f"    target_mag shape : {list(tm.shape)}  dtype={tm.dtype}")
        print(f"    mix_phase  shape : {list(mp.shape)}  dtype={mp.dtype}")
        assert mm.shape == tm.shape == mp.shape, "Shape mismatch between tensors"
        assert mm.dtype == torch.float32
    except Exception as e:
        errors.append(f"Dataset __getitem__: {e}")
        print(f"    ERROR: {e}")

# ── 3. DataLoader batch ───────────────────────────────────────────────────
batch = None
if ds is not None:
    print("\n[3] Creating DataLoader (batch_size=2)...")
    try:
        loaders = get_dataloaders(batch_size=2, chunk_duration=5.0, num_workers=0)
        train_loader = loaders["train"]
        print(f"    Train batches  : {len(train_loader)}")
        print(f"    Test  batches  : {len(loaders['test'])}")

        batch = next(iter(train_loader))
        print(f"    Batch mix_mag    : {list(batch['mix_mag'].shape)}")
        print(f"    Batch target_mag : {list(batch['target_mag'].shape)}")
        print(f"    Batch mix_phase  : {list(batch['mix_phase'].shape)}")

        B = batch["mix_mag"].shape[0]
        assert B <= 2, f"Batch size wrong: {B}"
    except Exception as e:
        errors.append(f"DataLoader: {e}")
        print(f"    ERROR: {e}")
        import traceback; traceback.print_exc()

# ── 4. Save side-by-side spectrograms ─────────────────────────────────────
print("\n[4] Saving phase2_spectrograms.png...")
try:
    proc = AudioProcessor()

    if batch is not None:
        mix_db    = proc.spectrogram_to_db(batch["mix_mag"][0])     # (1, F, T)
        target_db = proc.spectrogram_to_db(batch["target_mag"][0])
    elif sample is not None:
        mix_db    = proc.spectrogram_to_db(sample["mix_mag"])
        target_db = proc.spectrogram_to_db(sample["target_mag"])
    else:
        raise RuntimeError("No data available for plotting")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, spec, title in zip(
        axes,
        [mix_db[0].numpy(), target_db[0].numpy()],
        ["Mixture (dB)", "Vocals target (dB)"],
    ):
        im = ax.imshow(spec, aspect="auto", origin="lower", interpolation="none")
        ax.set_title(title)
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Freq bins")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    out_path = str(OUTPUTS / "phase2_spectrograms.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path}")
except Exception as e:
    errors.append(f"Plot: {e}")
    print(f"    ERROR: {e}")

# ── Summary ───────────────────────────────────────────────────────────────
print()
print("=" * 55)

checks = [
    ("MUSDBDataset init + __len__",  ds is not None),
    ("__getitem__ shapes correct",   sample is not None and not any("__getitem__" in e for e in errors)),
    ("DataLoader batch shapes",       batch is not None),
    ("Spectrogram plot saved",        (OUTPUTS / "phase2_spectrograms.png").exists()),
]

all_pass = True
for label, ok in checks:
    sym = "OK" if ok else "XX"
    print(f"  [{sym}] {label}")
    if not ok:
        all_pass = False

if errors:
    print()
    for e in errors:
        print(f"  ERROR: {e}")

print("=" * 55)
verdict = "Phase 2: PASS" if all_pass else f"Phase 2: FAIL -- {errors[0] if errors else 'see above'}"
print(f"  {verdict}")
print("=" * 55)
