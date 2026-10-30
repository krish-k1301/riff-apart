"""
Stem separation with Meta's HTDemucs 6-stem model (htdemucs_6s), which splits a
track into drums, bass, other, vocals, guitar and piano.

Weights (~52 MB) download to ~/.cache/torch/hub the first time the model loads.
The model then stays in memory, so only the first request pays for loading it.

Runs on an NVIDIA (CUDA) or Intel (XPU) GPU when PyTorch can see one, which is
several times faster than CPU, and falls back to an INT8-quantized CPU model.
"""

from __future__ import annotations

import io
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

MODEL_NAME = "htdemucs_6s"
SAMPLE_RATE = 44100  # HTDemucs works at 44.1 kHz

# Random-shift passes per quality setting. Each pass separates the whole track
# again at a small time offset and the passes are averaged: two passes clean up
# guitar noticeably but take twice as long.
SHIFTS = {"fast": 1, "high": 2}

# Fraction each chunk overlaps the next, cross-faded to hide the seams.
# 0.25 is Demucs' own default and needs a third fewer chunks than 0.5.
OVERLAP = 0.25

# (done_chunks, total_chunks). Raise from it to abort the separation.
ProgressCallback = Callable[[int, int], None]


class _ChunkCounter:
    """Stands in for the `tqdm` module inside demucs.apply and counts finished chunks.

    demucs only reports progress through a tqdm bar, so we swap this in while a
    separation runs.
    """

    def __init__(self, passes: int, on_progress: ProgressCallback):
        self.passes = passes
        self.on_progress = on_progress
        self.total = 0
        self.done = 0

    def tqdm(self, iterable, **_kwargs):
        items = list(iterable)
        if not self.total:
            # every pass (shift x sub-model) splits the track into the same chunks
            self.total = len(items) * self.passes
            self.on_progress(0, self.total)
        for item in items:
            yield item  # resumes once demucs has finished this chunk
            self.done += 1
            self.on_progress(min(self.done, self.total), self.total)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def _quantize(model: torch.nn.Module) -> torch.nn.Module:
    """INT8 dynamic quantization of the transformer's Linear layers (~15% faster on CPU).

    Dynamic quantization only covers Linear layers, so the convolutions stay float32.
    torch.ao.quantization is deprecated; if a future torch drops it, run unquantized.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    except AttributeError:
        return model


def load_audio(path: str) -> torch.Tensor:
    """Decode any audio file to a stereo float32 tensor [2, samples] at 44.1 kHz."""
    try:
        data, file_sr = sf.read(path, dtype="float32", always_2d=True)
        audio = data.T  # [channels, samples]
    except sf.LibsndfileError:
        # m4a/aac and other formats libsndfile can't read (librosa decodes them via ffmpeg)
        import librosa
        try:
            audio, file_sr = librosa.load(path, sr=None, mono=False)
        except Exception as exc:
            raise ValueError("Couldn’t read that file as audio. Try a wav, mp3 or flac.") from exc
        audio = np.atleast_2d(audio)

    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    if file_sr != SAMPLE_RATE:
        wav = AF.resample(wav, file_sr, SAMPLE_RATE)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)  # mono -> stereo
    return wav[:2]


class Separator:
    """Loads htdemucs_6s once, on first use, and runs one separation at a time."""

    def __init__(self):
        self.device = _pick_device()
        self._model = None
        # guards the shared model and the tqdm swap in demucs.apply
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            from demucs.pretrained import get_model
            model = get_model(MODEL_NAME).eval()
            # quantized layers only run on CPU
            self._model = _quantize(model) if self.device.type == "cpu" else model.to(self.device)
        return self._model

    def warm_up(self) -> None:
        """Load the model and separate a few seconds of silence.

        A GPU compiles its kernels on the first run (~25 s on Intel Arc), so
        doing it at startup keeps that cost off the first upload.
        """
        self.separate(torch.zeros(2, SAMPLE_RATE * 10), shifts=1)

    @torch.no_grad()
    def separate(
        self, mix: torch.Tensor, shifts: int, on_progress: ProgressCallback | None = None
    ) -> dict[str, np.ndarray]:
        """Split a [2, samples] mix into {stem: [samples, 2] float32 array}."""
        import demucs.apply as demucs_apply
        from demucs.apply import apply_model

        with self._lock:
            model = self._get_model()
            passes = shifts * len(getattr(model, "models", [model]))

            original_tqdm = demucs_apply.tqdm
            if on_progress:
                demucs_apply.tqdm = _ChunkCounter(passes, on_progress)
            try:
                sources = apply_model(
                    model, mix[None], device=self.device, shifts=shifts, overlap=OVERLAP,
                    progress=on_progress is not None,
                )  # [1, n_sources, 2, samples]
            finally:
                demucs_apply.tqdm = original_tqdm

        sources = sources[0].clamp(-1.0, 1.0).cpu().numpy()
        return {stem: np.ascontiguousarray(sources[i].T) for i, stem in enumerate(model.sources)}


def encode_mp3s(stems: dict[str, np.ndarray]) -> dict[str, bytes]:
    """Encode each stem as 320 kbps constant-bitrate MP3.

    The stems are encoded in parallel threads, since libsndfile releases the GIL
    while it encodes.
    """
    def encode(audio: np.ndarray) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="MP3", compression_level=0, bitrate_mode="CONSTANT")
        return buf.getvalue()

    with ThreadPoolExecutor(max_workers=len(stems)) as pool:
        return dict(zip(stems, pool.map(encode, stems.values())))
