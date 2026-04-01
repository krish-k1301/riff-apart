"""
Phase 3 smoke test — verifies model shapes and forward passes without any trained weights.

Run from the backend/ directory:
    python test_phase3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from app.models.unet import UNet
from app.models.classifier import InstrumentClassifier, STEMS

# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

print("=== U-Net ===")
unet = UNet(base_channels=32)
total_params = sum(p.numel() for p in unet.parameters())
print(f"Parameters: {total_params:,}")

# Typical batch from Phase 2: [B, 1, 1025, 431]
x = torch.randn(2, 1, 1025, 431)
mask = unet(x)
assert mask.shape == x.shape,          f"Mask shape mismatch: {mask.shape} != {x.shape}"
assert mask.min() >= 0.0,              f"Mask has negative values: {mask.min()}"
assert mask.max() <= 1.0,              f"Mask exceeds 1.0: {mask.max()}"
print(f"Input shape : {list(x.shape)}")
print(f"Mask shape  : {list(mask.shape)}")
print(f"Mask range  : [{mask.min():.4f}, {mask.max():.4f}]  (expected [0, 1])")
print("U-Net forward pass — PASS\n")

# ---------------------------------------------------------------------------
# Instrument Classifier
# ---------------------------------------------------------------------------

print("=== InstrumentClassifier ===")
clf = InstrumentClassifier(base_channels=32)
total_params = sum(p.numel() for p in clf.parameters())
print(f"Parameters: {total_params:,}")

logits = clf(x)
probs  = clf.predict_proba(x)

assert logits.shape == (2, 4), f"Logits shape mismatch: {logits.shape}"
assert probs.shape  == (2, 4), f"Proba shape mismatch: {probs.shape}"
assert probs.min() >= 0.0 and probs.max() <= 1.0, "Probabilities out of [0, 1]"
print(f"Input shape  : {list(x.shape)}")
print(f"Logits shape : {list(logits.shape)}")
print(f"Stems        : {STEMS}")
print(f"Sample probs : {[round(p, 4) for p in probs[0].tolist()]}")
print("Classifier forward pass — PASS\n")

# ---------------------------------------------------------------------------
# Mask application (separator logic without loading weights)
# ---------------------------------------------------------------------------

print("=== Mask application ===")
mix_mag    = torch.rand(2, 1, 1025, 431)
target_mag = torch.rand(2, 1, 1025, 431)

mask       = unet(mix_mag)
pred       = mask * mix_mag
loss       = torch.nn.functional.l1_loss(pred, target_mag)
loss.backward()

print(f"L1 loss (random weights): {loss.item():.6f}")
print(f"Gradient check on first conv weight: {unet.enc1[0].weight.grad is not None}")
print("Backward pass — PASS\n")

print("=== All Phase 3 checks passed ===")
