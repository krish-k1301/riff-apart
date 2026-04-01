from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_spectrogram(
    spectrogram: torch.Tensor,
    title: str = "Spectrogram",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a 2-D spectrogram tensor (freq_bins, time_frames).
    If a channel dimension is present, the first channel is displayed.

    Args:
        spectrogram: Tensor of shape (freq_bins, time_frames) or
                     (channels, freq_bins, time_frames).
        title: Plot title.
        save_path: If given, the figure is saved here instead of shown.
    """
    spec = spectrogram.detach().cpu()

    # Collapse channel dim if present
    if spec.ndim == 3:
        spec = spec[0]

    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(
        spec.numpy(),
        aspect="auto",
        origin="lower",
        interpolation="none",
    )
    ax.set_title(title)
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Frequency bins")
    fig.colorbar(img, ax=ax)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_waveform(
    waveform: torch.Tensor,
    sr: int,
    title: str = "Waveform",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot an audio waveform tensor.

    Args:
        waveform: Tensor of shape (channels, samples) or (samples,).
        sr: Sample rate (used to build a time axis in seconds).
        title: Plot title.
        save_path: If given, the figure is saved here instead of shown.
    """
    wav = waveform.detach().cpu()

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    num_channels, num_samples = wav.shape
    time_axis = np.linspace(0, num_samples / sr, num_samples)

    fig, axes = plt.subplots(num_channels, 1, figsize=(10, 2 * num_channels), squeeze=False)
    fig.suptitle(title)

    for ch, ax in enumerate(axes[:, 0]):
        ax.plot(time_axis, wav[ch].numpy(), linewidth=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Ch {ch}")
        ax.set_ylim(-1.1, 1.1)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
