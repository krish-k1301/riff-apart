"""
Phase 1 test script — Audio Processing Module

Generates a synthetic 440 Hz sine wave, runs it through the full
loader -> STFT -> magnitude/phase -> iSTFT -> save pipeline, and
verifies that the reconstruction error is near zero.
"""

import sys
from pathlib import Path

import torch
import numpy as np

# Make sure the backend package is importable when run from backend/
sys.path.insert(0, str(Path(__file__).parent))

from app.audio.loader import load_audio, save_audio
from app.audio.processor import AudioProcessor
from app.audio.utils import plot_spectrogram, plot_waveform

OUTPUTS = Path(__file__).parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

SR = 44100
DURATION = 3.0  # seconds
FREQ = 440.0    # Hz (A4)


# ── 1. Generate synthetic sine wave ────────────────────────────────────────
t = torch.linspace(0, DURATION, int(SR * DURATION))
sine = torch.sin(2 * torch.pi * FREQ * t).unsqueeze(0)  # (1, samples)
print(f"[1] Synthetic tone  | shape: {sine.shape}  sr: {SR}")

# Save original to disk then reload through loader.py
orig_path = str(OUTPUTS / "original_440hz.wav")
save_audio(sine, orig_path, sr=SR)
print(f"    Saved original -> {orig_path}")

waveform, sr = load_audio(orig_path, sr=SR, mono=True)
print(f"    Loaded back      | shape: {waveform.shape}  sr: {sr}")


# ── 2. STFT ────────────────────────────────────────────────────────────────
proc = AudioProcessor(n_fft=2048, hop_length=512, win_length=2048)

stft = proc.compute_stft(waveform)
print(f"\n[2] STFT             | shape: {stft.shape}  dtype: {stft.dtype}")

magnitude = proc.compute_magnitude(stft)
phase     = proc.compute_phase(stft)
print(f"    Magnitude        | shape: {magnitude.shape}  min: {magnitude.min():.4f}  max: {magnitude.max():.4f}")
print(f"    Phase            | shape: {phase.shape}     min: {phase.min():.4f}      max: {phase.max():.4f}")


# ── 3. Mel spectrogram ─────────────────────────────────────────────────────
mel = proc.compute_mel_spectrogram(waveform, n_mels=128, sr=sr)
print(f"\n[3] Mel spectrogram  | shape: {mel.shape}")


# ── 4. Reconstruct ─────────────────────────────────────────────────────────
reconstructed = proc.inverse_stft(magnitude, phase)
print(f"\n[4] Reconstructed    | shape: {reconstructed.shape}")


# ── 5. Trim both to the shorter length (iSTFT frames may not cover all samples) ──
n = min(waveform.shape[-1], reconstructed.shape[-1])
waveform = waveform[..., :n]
reconstructed = reconstructed[..., :n]

mse = torch.mean((waveform - reconstructed) ** 2).item()
max_err = (waveform - reconstructed).abs().max().item()
print(f"\n[5] Reconstruction error")
print(f"    MSE      : {mse:.2e}  {'PASS OK' if mse < 1e-6 else 'FAIL FAIL (expected < 1e-6)'}")
print(f"    Max abs  : {max_err:.2e}")


# ── 6. Save reconstructed audio ───────────────────────────────────────────
recon_path = str(OUTPUTS / "reconstructed_440hz.wav")
save_audio(reconstructed, recon_path, sr=sr)
print(f"\n[6] Saved reconstructed -> {recon_path}")


# ── 7. Plots ───────────────────────────────────────────────────────────────
plot_waveform(waveform, sr, title="Original 440 Hz Sine Wave",
              save_path=str(OUTPUTS / "waveform_original.png"))
print(f"[7] Saved waveform plot -> {OUTPUTS / 'waveform_original.png'}")

mag_db = proc.spectrogram_to_db(magnitude)
plot_spectrogram(mag_db, title="Magnitude Spectrogram (dB)",
                 save_path=str(OUTPUTS / "spectrogram_magnitude_db.png"))
print(f"    Saved spectrogram plot -> {OUTPUTS / 'spectrogram_magnitude_db.png'}")

mel_db = proc.spectrogram_to_db(mel)
plot_spectrogram(mel_db, title="Mel Spectrogram (dB)",
                 save_path=str(OUTPUTS / "spectrogram_mel_db.png"))
print(f"    Saved mel plot -> {OUTPUTS / 'spectrogram_mel_db.png'}")


# ── 8. Summary ─────────────────────────────────────────────────────────────
print("\n-- Summary ----------------------------------------------------------")
print(f"  Input waveform    : {list(waveform.shape)}")
print(f"  STFT complex      : {list(stft.shape)}")
print(f"  Magnitude         : {list(magnitude.shape)}")
print(f"  Phase             : {list(phase.shape)}")
print(f"  Mel spectrogram   : {list(mel.shape)}")
print(f"  Reconstructed     : {list(reconstructed.shape)}")
print(f"  MSE               : {mse:.2e}")
print("---------------------------------------------------------------------")
