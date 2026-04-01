import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
from pathlib import Path


def load_audio(filepath: str, sr: int = 44100, mono: bool = True) -> tuple[torch.Tensor, int]:
    """
    Load an audio file (WAV, MP3, FLAC, etc.) and resample to target sample rate.

    Args:
        filepath: Path to the audio file.
        sr: Target sample rate.
        mono: If True, mix down to a single channel.

    Returns:
        (waveform, sample_rate) where waveform has shape (channels, samples).
    """
    filepath = str(filepath)
    try:
        # soundfile handles WAV/FLAC reliably without needing torchcodec
        data, orig_sr = sf.read(filepath, always_2d=True)  # (samples, channels)
        waveform = torch.from_numpy(data.T.astype(np.float32))  # (channels, samples)
    except Exception:
        # Fallback to librosa for MP3 and other formats soundfile can't read
        import librosa
        y, orig_sr = librosa.load(filepath, sr=None, mono=False)
        if y.ndim == 1:
            y = y[np.newaxis, :]
        waveform = torch.from_numpy(y.astype(np.float32))

    # Resample if needed
    if orig_sr != sr:
        resampler = T.Resample(orig_freq=orig_sr, new_freq=sr)
        waveform = resampler(waveform)

    # Convert to mono by averaging channels
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform, sr


def save_audio(tensor: torch.Tensor, filepath: str, sr: int = 44100) -> None:
    """
    Save a torch tensor as a WAV file.

    Args:
        tensor: Audio tensor of shape (channels, samples) or (samples,).
        filepath: Destination file path (should end in .wav).
        sr: Sample rate.
    """
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    # Clamp to [-1, 1] to avoid clipping artifacts
    tensor = tensor.clamp(-1.0, 1.0)

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    # torchaudio 2.11+ requires torchcodec for saving; use soundfile instead
    audio_np = tensor.numpy().T  # soundfile expects (samples, channels)
    sf.write(filepath, audio_np, sr, subtype="PCM_16")
