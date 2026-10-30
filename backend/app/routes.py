"""
Separation jobs with live progress.

POST /api/jobs saves the upload and separates it in a background thread. The
frontend polls GET /api/jobs/{id} for progress and an ETA, then downloads the
stems as a ZIP of MP3s from GET /api/jobs/{id}/result.
"""

import io
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app.separator import SAMPLE_RATE, SHIFTS, Separator, encode_mp3s, load_audio

router = APIRouter(prefix="/api")
separator = Separator()

JOB_TTL_SECONDS = 15 * 60
_jobs: dict[str, "Job"] = {}
_jobs_lock = threading.Lock()

# Compute seconds per second of audio, learned from the last finished job so the
# ETA is sensible before the first chunk completes. Seeded with rough guesses.
_secs_per_audio_sec = {
    "separate": 0.6 if separator.device.type == "cpu" else 0.15,  # per shift pass
    "encode": 0.02,  # all six stems, encoded in parallel
}


class _Cancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    shifts: int
    status: str = "queued"  # queued | loading | separating | packaging | done | error | cancelled
    done: int = 0
    total: int = 0
    audio_seconds: float | None = None
    created: float = field(default_factory=time.monotonic)
    sep_started: float | None = None
    pack_started: float | None = None
    finished: float | None = None
    error: str | None = None
    result: bytes | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def eta_seconds(self) -> float | None:
        if self.status in ("done", "error", "cancelled"):
            return 0.0
        now = time.monotonic()
        encode = (self.audio_seconds or 0.0) * _secs_per_audio_sec["encode"] + 0.5
        if self.status == "packaging":
            return max(encode - (now - self.pack_started), 0.5)

        elapsed = now - self.sep_started if self.sep_started else 0.0
        # prior: how long tracks this length took before
        prior = None
        if self.audio_seconds:
            expected = self.audio_seconds * self.shifts * _secs_per_audio_sec["separate"]
            prior = max(expected - elapsed, 1.0)
        # measured: chunk rate so far (the first chunk includes warm-up, so it reads high)
        measured = None
        if self.sep_started and self.done > 0:
            measured = elapsed / self.done * (self.total - self.done)
        if prior is None and measured is None:
            return None
        if prior is None or measured is None:
            return (prior if measured is None else measured) + encode
        trust = self.done / self.total  # lean on measurements as they accumulate
        return (1 - trust) * prior + trust * measured + encode

    def to_dict(self) -> dict:
        eta = self.eta_seconds()
        return {
            "id": self.id,
            "status": self.status,
            "progress": round(self.done / self.total, 4) if self.total else 0.0,
            "chunks_done": self.done,
            "chunks_total": self.total,
            "audio_seconds": self.audio_seconds,
            "elapsed_seconds": round(time.monotonic() - self.created, 1),
            "eta_seconds": None if eta is None else round(eta, 1),
            "error": self.error,
        }


def _run_job(job: Job, input_path: str, tmpdir: str) -> None:
    def on_progress(done: int, total: int) -> None:
        if job.cancel.is_set():
            raise _Cancelled()
        if job.sep_started is None:
            job.sep_started = time.monotonic()
            job.status = "separating"
        job.done, job.total = done, total

    try:
        job.status = "loading"  # decoding audio, plus loading the model on the first run
        mix = load_audio(input_path)
        if mix.shape[-1] == 0:
            raise ValueError("That file has no audio in it.")
        job.audio_seconds = mix.shape[-1] / SAMPLE_RATE
        stems = separator.separate(mix, job.shifts, on_progress)
        _secs_per_audio_sec["separate"] = (time.monotonic() - job.sep_started) / (job.audio_seconds * job.shifts)

        if job.cancel.is_set():
            raise _Cancelled()
        job.status = "packaging"
        job.pack_started = time.monotonic()
        mp3s = encode_mp3s(stems)
        _secs_per_audio_sec["encode"] = (time.monotonic() - job.pack_started) / job.audio_seconds

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_STORED) as zf:  # MP3s don't compress further
            for stem, data in mp3s.items():
                zf.writestr(f"{stem}.mp3", data)
        job.result = zip_buffer.getvalue()
        job.status = "done"
    except _Cancelled:
        job.status = "cancelled"
    except Exception as exc:  # surfaced to the client via the status endpoint
        job.status = "error"
        job.error = str(exc) or "Separation failed"
    finally:
        job.finished = time.monotonic()
        shutil.rmtree(tmpdir, ignore_errors=True)


def _purge_old_jobs() -> None:
    cutoff = time.monotonic() - JOB_TTL_SECONDS
    with _jobs_lock:
        for job_id in [j.id for j in _jobs.values() if j.finished and j.finished < cutoff]:
            _jobs.pop(job_id, None)


def _find_job(job_id: str) -> Job:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found. It may have expired, so upload the track again.")
    return job


@router.post("/jobs", status_code=202)
def create_job(
    file: UploadFile = File(...),
    quality: Literal["fast", "high"] = Query("fast", description="fast: 1 pass, high: 2 passes (cleaner, 2x slower)"),
):
    """Start separating all 6 stems in the background. Poll /api/jobs/{id} for progress."""
    _purge_old_jobs()
    # keep the extension so the decoder can tell what format the file is
    suffix = Path(file.filename or "").suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,5}", suffix):
        suffix = ""
    tmpdir = tempfile.mkdtemp(prefix="riffapart-")
    input_path = str(Path(tmpdir) / f"input{suffix}")
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job = Job(id=uuid.uuid4().hex, shifts=SHIFTS[quality])
    with _jobs_lock:
        _jobs[job.id] = job
    threading.Thread(target=_run_job, args=(job, input_path, tmpdir), daemon=True).start()
    return job.to_dict()


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    return _find_job(job_id).to_dict()


@router.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    job = _find_job(job_id)
    if job.status != "done" or job.result is None:
        raise HTTPException(status_code=409, detail=f"Job is {job.status}, not done yet.")
    with _jobs_lock:
        _jobs.pop(job_id, None)
    return Response(
        content=job.result,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="stems.zip"'},
    )


@router.delete("/jobs/{job_id}", status_code=204)
def cancel_job(job_id: str):
    job = _find_job(job_id)
    job.cancel.set()
    if job.finished:
        with _jobs_lock:
            _jobs.pop(job_id, None)
    return Response(status_code=204)
