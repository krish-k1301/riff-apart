import os
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

INSTRUMENTS = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]
INSTRUMENT_NAMES = {
    "cel": "cello",
    "cla": "clarinet",
    "flu": "flute",
    "gac": "acoustic guitar",
    "gel": "electric guitar",
    "org": "organ",
    "pia": "piano",
    "sax": "saxophone",
    "tru": "trumpet",
    "vio": "violin",
    "voi": "voice",
}
NUM_CLASSES = len(INSTRUMENTS)


class IRMASDataset(Dataset):
    """
    PyTorch Dataset for IRMAS training data.

    Folder structure expected:
        root/
            cel/  *.wav
            cla/  *.wav
            ...

    Each clip is ~3 seconds at 44100 Hz (mono).
    Each clip belongs to exactly one instrument class (single-label).

    Returns:
        mel_spec : float32 tensor [1, n_mels, time_frames]
        label    : int64 tensor  [NUM_CLASSES]  one-hot encoded
    """

    def __init__(
        self,
        root: str,
        sr: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        clip_duration: float = 3.0,
    ):
        self.root = root
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.clip_samples = int(clip_duration * sr)

        self.samples = []  # list of (filepath, class_index)
        for idx, instrument in enumerate(INSTRUMENTS):
            folder = os.path.join(root, instrument)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(".wav"):
                    self.samples.append((os.path.join(folder, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, class_idx = self.samples[idx]

        # Load audio
        audio, file_sr = sf.read(filepath, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)  # stereo → mono

        # Resample if needed (simple linear interp — librosa not required)
        if file_sr != self.sr:
            target_len = int(len(audio) * self.sr / file_sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, target_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)

        # Pad or crop to fixed length
        if len(audio) < self.clip_samples:
            audio = np.pad(audio, (0, self.clip_samples - len(audio)))
        else:
            audio = audio[: self.clip_samples]

        # Mel spectrogram via numpy/torch (no torchaudio backend needed)
        mel_spec = self._mel_spectrogram(audio)

        # One-hot label
        label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        label[class_idx] = 1.0

        return mel_spec, label

    def _mel_spectrogram(self, audio: np.ndarray) -> torch.Tensor:
        """Compute log-mel spectrogram using torch STFT."""
        waveform = torch.from_numpy(audio).unsqueeze(0)  # [1, T]

        window = torch.hann_window(self.n_fft)
        stft = torch.stft(
            waveform.squeeze(0),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
        )
        magnitude = stft.abs()  # [freq_bins, time_frames]

        # Build mel filterbank
        mel_fb = self._mel_filterbank(self.n_mels, self.n_fft, self.sr)  # [n_mels, freq_bins]
        mel = torch.matmul(mel_fb, magnitude)  # [n_mels, time_frames]

        # Log compression
        mel = torch.log1p(mel * 1000.0)

        return mel.unsqueeze(0).float()  # [1, n_mels, time_frames]

    @staticmethod
    def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> torch.Tensor:
        """Create a mel filterbank matrix [n_mels, n_fft//2 + 1]."""
        fmin, fmax = 0.0, sr / 2.0
        freq_bins = n_fft // 2 + 1

        def hz_to_mel(f):
            return 2595.0 * np.log10(1.0 + f / 700.0)

        def mel_to_hz(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        fb = np.zeros((n_mels, freq_bins), dtype=np.float32)
        for m in range(1, n_mels + 1):
            f_left, f_center, f_right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
            for k in range(f_left, f_center):
                if f_center != f_left:
                    fb[m - 1, k] = (k - f_left) / (f_center - f_left)
            for k in range(f_center, f_right):
                if f_right != f_center:
                    fb[m - 1, k] = (f_right - k) / (f_right - f_center)

        return torch.from_numpy(fb)
