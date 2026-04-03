"""
Demucs-based inference pipeline.

Uses HTDemucs 6-stem model (htdemucs_6s) which separates:
    drums, bass, other, vocals, guitar, piano

The model was trained by Meta Research on a large curated dataset using a
hybrid transformer architecture. It runs fully on CPU (or GPU if available).
Weights are downloaded automatically to ~/.cache/torch/hub on first use (~52 MB).

Usage:
    pipeline = DemucsPipeline()
    # Single stem:
    pipeline.run_single("song.wav", "vocals_out.wav", stem="vocals")
    # All stems:
    paths = pipeline.run_all("song.wav", output_dir="outputs/stems/")
"""

from __future__ import annotations

import torch
import torchaudio.transforms as T
from pathlib import Path

from app.audio.loader import load_audio, save_audio

MODEL_NAME = "htdemucs_6s"
STEMS = ["drums", "bass", "other", "vocals", "guitar", "piano"]


class DemucsPipeline:
    """
    Wraps htdemucs_6s for single-stem or all-stem separation.

    The model is loaded once and reused across calls.  Loading is deferred to
    the first call so the API starts up instantly even if the weights are not
    yet cached.
    """

    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._model = None   # lazy-loaded

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is not None:
            return
        from demucs.pretrained import get_model
        self._model = get_model(MODEL_NAME)
        self._model.to(self.device).eval()

        # INT8 dynamic quantization — CPU only (quantized ops don't run on CUDA)
        if self.device.type == "cpu":
            self._model = torch.quantization.quantize_dynamic(
                self._model,
                {torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d},
                dtype=torch.qint8,
            )


    def _load_stereo(self, input_path: str) -> torch.Tensor:
        """
        Load audio as a stereo float32 tensor [2, samples] at the model's
        sample rate (44100 Hz).  Mono files are duplicated to stereo.

        Uses soundfile/librosa (via load_audio) to avoid the torchaudio
        torchcodec dependency that is unavailable on Windows + Python 3.13.
        load_audio always returns mono [1, samples], so we load both channels
        manually via soundfile when the file is natively stereo.
        """
        import soundfile as sf
        import numpy as np

        self._load_model()
        model_sr = self._model.samplerate

        try:
            data, file_sr = sf.read(input_path, dtype="float32", always_2d=True)
            # data: [samples, channels]
            waveform = torch.from_numpy(data.T)  # [channels, samples]
        except Exception:
            # Fallback for MP3 and other formats soundfile can't handle
            import librosa
            data, file_sr = librosa.load(input_path, sr=None, mono=False)
            if data.ndim == 1:
                data = data[np.newaxis, :]
            waveform = torch.from_numpy(data).float()

        # Resample if needed
        if file_sr != model_sr:
            waveform = T.Resample(orig_freq=file_sr, new_freq=model_sr)(waveform)

        # Ensure exactly 2 channels
        if waveform.shape[0] == 1:
            waveform = waveform.expand(2, -1).clone()  # mono → stereo
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        return waveform.float()

    @torch.no_grad()
    def _separate(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Run htdemucs_6s on a stereo waveform and return all stems.

        Args:
            waveform: [2, samples] float32 at 44100 Hz

        Returns:
            dict of stem_name → [2, samples] tensor
        """
        from demucs.apply import apply_model

        mix = waveform.unsqueeze(0).to(self.device)  # [1, 2, samples]

        # shifts=1: single inference (no random-shift ensemble — faster, still good)
        # overlap=0.25: 25% chunk overlap for smooth boundaries (Demucs handles this internally)
        sources = apply_model(
            self._model,
            mix,
            device=self.device,
            shifts=2,   # random-shift ensemble: significantly improves guitar quality
            overlap=0.5,
            progress=False,
        )  # [1, n_sources, 2, samples]

        sources = sources.squeeze(0).cpu()   # [n_sources, 2, samples]
        return {
            stem: sources[i]
            for i, stem in enumerate(self._model.sources)
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_single(self, input_path: str, output_path: str, stem: str) -> str:
        """
        Separate one stem from input_path and write it to output_path.

        Args:
            input_path:  Source audio file (any format torchaudio can read).
            output_path: Destination WAV path.
            stem:        One of: drums, bass, other, vocals, guitar, piano.

        Returns:
            output_path (for convenience).
        """
        if stem not in STEMS:
            raise ValueError(f"stem must be one of {STEMS}, got '{stem}'")

        waveform = self._load_stereo(input_path)
        stems = self._separate(waveform)
        save_audio(stems[stem], output_path, sr=self._model.samplerate)
        return output_path

    def run_all(self, input_path: str, output_dir: str) -> dict[str, str]:
        """
        Separate all 6 stems and write each to output_dir/{stem}.wav.

        Returns:
            dict of stem_name → output file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        waveform = self._load_stereo(input_path)
        stems = self._separate(waveform)

        output_paths: dict[str, str] = {}
        for stem_name, audio in stems.items():
            out_path = str(output_dir / f"{stem_name}.wav")
            save_audio(audio, out_path, sr=self._model.samplerate)
            output_paths[stem_name] = out_path

        return output_paths
