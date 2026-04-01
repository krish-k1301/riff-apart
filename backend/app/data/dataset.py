import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset

# stempeg checks for ffmpeg at import time via shutil.which.
# On Windows, winget installs to a path not yet in the process's PATH
# (until the shell is restarted). Find and inject it dynamically.
def _ensure_ffmpeg_on_path():
    import shutil
    if shutil.which("ffmpeg"):
        return
    candidates = glob.glob(
        os.path.expanduser(
            r"~\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg-*\bin"
        )
    )
    if candidates:
        os.environ["PATH"] = candidates[0] + os.pathsep + os.environ.get("PATH", "")

_ensure_ffmpeg_on_path()

import musdb

from app.audio.processor import AudioProcessor


class MUSDBDataset(Dataset):
    """
    Wraps the MUSDB18 dataset.  Each __getitem__ returns a random chunk
    from the selected track as magnitude spectrograms ready for a U-Net.

    Returns
    -------
    dict with keys:
        mix_mag    : (1, freq_bins, time_frames)  float32
        target_mag : (1, freq_bins, time_frames)  float32
        mix_phase  : (1, freq_bins, time_frames)  float32
    """

    def __init__(
        self,
        root=None,
        split="train",
        target="vocals",
        chunk_duration=5.0,
        sr=44100,
        n_fft=2048,
        hop_length=512,
        download=True,
    ):
        self.split = split
        self.target = target
        self.chunk_duration = chunk_duration
        self.sr = sr
        self.chunk_samples = int(sr * chunk_duration)

        self.processor = AudioProcessor(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft
        )

        if root is None:
            self.db = musdb.DB(download=download, subsets=[split])
        else:
            self.db = musdb.DB(root=root, subsets=[split])

        self.tracks = self.db.tracks

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]

        # musdb uses sample rate from the file; normalise to self.sr via
        # simple resampling after converting to tensor
        track_sr = track.rate  # usually 44100

        total_samples = track.audio.shape[0]  # (samples, channels)
        chunk_samples = int(track_sr * self.chunk_duration)

        # ── random chunk start ────────────────────────────────────────
        if total_samples > chunk_samples:
            start = random.randint(0, total_samples - chunk_samples)
        else:
            start = 0

        end = start + chunk_samples

        # ── load mixture and target stem ──────────────────────────────
        mix_np = track.audio[start:end]           # (samples, 2)
        target_np = track.targets[self.target].audio[start:end]

        mix_np    = self._to_mono_float(mix_np,    chunk_samples)
        target_np = self._to_mono_float(target_np, chunk_samples)

        mix_t    = torch.from_numpy(mix_np).unsqueeze(0)     # (1, samples)
        target_t = torch.from_numpy(target_np).unsqueeze(0)

        # ── random gain augmentation (same factor for both) ───────────
        gain = random.uniform(0.8, 1.2)
        mix_t    = mix_t    * gain
        target_t = target_t * gain

        # ── spectrograms ──────────────────────────────────────────────
        mix_stft    = self.processor.compute_stft(mix_t)
        target_stft = self.processor.compute_stft(target_t)

        mix_mag    = self.processor.compute_magnitude(mix_stft)    # (1, F, T)
        mix_phase  = self.processor.compute_phase(mix_stft)        # (1, F, T)
        target_mag = self.processor.compute_magnitude(target_stft) # (1, F, T)

        return {
            "mix_mag":    mix_mag.float(),
            "target_mag": target_mag.float(),
            "mix_phase":  mix_phase.float(),
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_mono_float(audio_np: np.ndarray, target_len: int) -> np.ndarray:
        """Convert (samples, channels) numpy array to mono float32, zero-padded."""
        if audio_np.ndim == 2:
            audio_np = audio_np.mean(axis=1)
        audio_np = audio_np.astype(np.float32)

        # zero-pad if track shorter than chunk
        if len(audio_np) < target_len:
            pad = np.zeros(target_len - len(audio_np), dtype=np.float32)
            audio_np = np.concatenate([audio_np, pad])

        return audio_np[:target_len]
