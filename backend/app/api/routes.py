import tempfile
from pathlib import Path

import torch
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app.audio.loader import load_audio
from app.audio.processor import AudioProcessor
from app.models.classifier import InstrumentClassifier, STEMS
from app.pipeline.inference import InferencePipeline

router = APIRouter(prefix="/api")

VALID_TARGETS = {"vocals", "drums", "bass", "other"}
_CHECKPOINTS_DIR = Path(__file__).parent.parent.parent / "checkpoints"
_SR = 44100
_N_FFT = 2048
_HOP_LENGTH = 512
_WIN_LENGTH = 2048


def _unet_path(target: str) -> Path:
    path = _CHECKPOINTS_DIR / f"unet_{target}_best.pt"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for target '{target}'. Train it first with train.py.",
        )
    return path


@router.post("/separate")
async def separate(
    file: UploadFile = File(...),
    target: str = Query(..., description="vocals | drums | bass | other"),
):
    """
    Upload an audio file and receive the separated stem as a WAV download.
    """
    if target not in VALID_TARGETS:
        raise HTTPException(status_code=400, detail=f"target must be one of {sorted(VALID_TARGETS)}")

    model_path = str(_unet_path(target))

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = str(Path(tmpdir) / "input.wav")
        output_path = str(Path(tmpdir) / f"{target}.wav")

        with open(input_path, "wb") as f:
            f.write(await file.read())

        pipeline = InferencePipeline(
            model_path=model_path,
            device="cpu",
            sr=_SR,
            n_fft=_N_FFT,
            hop_length=_HOP_LENGTH,
            win_length=_WIN_LENGTH,
        )
        pipeline.run(input_path, output_path)

        with open(output_path, "rb") as f:
            audio_bytes = f.read()

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{target}_separated.wav"'},
    )


@router.post("/classify")
async def classify(file: UploadFile = File(...)):
    """
    Upload an audio file and receive per-stem activity probabilities.

    Returns a JSON dict: {"vocals": 0.92, "drums": 0.87, "bass": 0.76, "other": 0.41}
    """
    classifier_path = _CHECKPOINTS_DIR / "classifier_best.pt"
    if not classifier_path.exists():
        raise HTTPException(status_code=404, detail="Classifier model not found.")

    processor = AudioProcessor(n_fft=_N_FFT, hop_length=_HOP_LENGTH, win_length=_WIN_LENGTH)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        waveform, _ = load_audio(tmp_path, sr=_SR, mono=True)
        # Use first 5 seconds for classification
        waveform = waveform[:, : 5 * _SR]

        stft = processor.compute_stft(waveform)
        mix_mag = processor.compute_magnitude(stft).unsqueeze(0)  # [1, 1, F, T]

        model = InstrumentClassifier()
        checkpoint = torch.load(classifier_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()

        probs = model.predict_proba(mix_mag).squeeze(0).tolist()
        return {stem: round(prob, 4) for stem, prob in zip(STEMS, probs)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)
