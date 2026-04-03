import tempfile
from pathlib import Path

import torch
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app.audio.loader import load_audio
from app.audio.processor import AudioProcessor
from app.data.irmas_dataset import INSTRUMENTS, INSTRUMENT_NAMES
from app.models.classifier import InstrumentClassifier
from app.pipeline.demucs_pipeline import DemucsPipeline, STEMS as DEMUCS_STEMS

router = APIRouter(prefix="/api")

VALID_TARGETS = set(DEMUCS_STEMS)  # drums, bass, other, vocals, guitar, piano
_CHECKPOINTS_DIR = Path(__file__).parent.parent.parent / "checkpoints"

# Shared pipeline instance — model is loaded lazily on first request
_demucs: DemucsPipeline | None = None


def _get_demucs() -> DemucsPipeline:
    global _demucs
    if _demucs is None:
        _demucs = DemucsPipeline(device="cpu")
    return _demucs


@router.post("/separate")
async def separate(
    file: UploadFile = File(...),
    target: str = Query(..., description="vocals | drums | bass | guitar | piano | other"),
):
    """
    Upload an audio file and receive the separated stem as a WAV download.
    Powered by HTDemucs 6-stem (transformer model by Meta Research).
    Supports: vocals, drums, bass, guitar, piano, other.
    """
    if target not in VALID_TARGETS:
        raise HTTPException(status_code=400, detail=f"target must be one of {sorted(VALID_TARGETS)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = str(Path(tmpdir) / "input.wav")
        output_path = str(Path(tmpdir) / f"{target}.wav")

        with open(input_path, "wb") as f:
            f.write(await file.read())

        _get_demucs().run_single(input_path, output_path, stem=target)

        with open(output_path, "rb") as f:
            audio_bytes = f.read()

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{target}_separated.wav"'},
    )


@router.post("/separate_all")
async def separate_all(file: UploadFile = File(...)):
    """
    Upload an audio file and receive all 6 stems in a ZIP archive.
    Stems: vocals, drums, bass, guitar, piano, other.
    Powered by HTDemucs 6-stem (transformer model by Meta Research).
    """
    import io
    import zipfile

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = str(Path(tmpdir) / "input.wav")
        output_dir = str(Path(tmpdir) / "stems")

        with open(input_path, "wb") as f:
            f.write(await file.read())

        output_paths = _get_demucs().run_all(input_path, output_dir)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for stem, path in output_paths.items():
                zf.write(path, arcname=f"{stem}.wav")
        zip_bytes = zip_buffer.getvalue()

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="stems.zip"'},
    )


@router.post("/classify")
async def classify(file: UploadFile = File(...)):
    """
    Upload an audio file and receive per-instrument probabilities using IRMAS classifier.

    Returns a JSON dict: {"cello": 0.02, "clarinet": 0.05, ..., "voice": 0.91}
    """
    classifier_path = _CHECKPOINTS_DIR / "classifier_best.pt"
    if not classifier_path.exists():
        raise HTTPException(status_code=404, detail="Classifier model not found. Train it first with train_classifier.py.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        from app.data.irmas_dataset import IRMASDataset
        import numpy as np
        import soundfile as sf

        # Load and preprocess audio the same way IRMASDataset does
        audio, file_sr = sf.read(tmp_path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)

        target_sr = 22050
        if file_sr != target_sr:
            target_len = int(len(audio) * target_sr / file_sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, target_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)

        # Use first 3 seconds (same as training)
        clip_samples = int(3.0 * target_sr)
        if len(audio) < clip_samples:
            audio = np.pad(audio, (0, clip_samples - len(audio)))
        else:
            audio = audio[:clip_samples]

        dummy_ds = IRMASDataset.__new__(IRMASDataset)
        dummy_ds.sr = target_sr
        dummy_ds.n_fft = 1024
        dummy_ds.hop_length = 512
        dummy_ds.n_mels = 128
        mel = dummy_ds._mel_spectrogram(audio).unsqueeze(0)  # [1, 1, n_mels, T]

        model = InstrumentClassifier()
        checkpoint = torch.load(classifier_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()

        probs = model.predict_proba(mel).squeeze(0).tolist()
        return {INSTRUMENT_NAMES[inst]: round(prob, 4) for inst, prob in zip(INSTRUMENTS, probs)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)
