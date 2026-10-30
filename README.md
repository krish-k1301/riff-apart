# Riff Apart

**built by a guitarist who got tired of bad stems**

[Live site](https://riff-apart.vercel.app/)

A full-stack music source separation app powered by Meta's HTDemucs. Upload any song, get 6 isolated stems i.e. vocals, drums, bass, guitar, piano, other. All downloadable, no login required.

## How to use

1. Open [riff-apart.vercel.app](https://riff-apart.vercel.app/).
2. Pick **Fast** (default) or **High** (two passes, cleaner guitar, about twice as slow) quality.
3. Upload a song from your device.
4. Wait while Riff Apart separates it into vocals, drums, bass, guitar, piano, and other.
5. Play stems individually or together in the mixer, with mute/solo per stem.
6. Download any stem as a 320 kbps MP3.

## Stack

- **Backend**: Python · FastAPI · PyTorch · Demucs (HTDemucs 6s) · soundfile
- **Frontend**: React · Vite
- **Model**: Meta's HTDemucs 6-stem - trained on a large curated dataset, hybrid transformer architecture

---

## Features

- 6-stem separation: vocals, drums, bass, guitar, piano, other
- Full song, no time limits, no login
- High / Fast quality modes
- Stems delivered as 320 kbps CBR MP3
- Live progress and ETA while separating
- Runs on an NVIDIA (CUDA) or Intel (XPU) GPU when available, otherwise an INT8-quantized CPU model

---

## Quick start

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

m4a/aac uploads need `ffmpeg` on the PATH.

## How it works

1. `POST /api/jobs?quality=high|fast` saves the upload and starts a background job.
2. The frontend polls `GET /api/jobs/{id}` for progress and an ETA.
3. The job decodes the audio to 44.1 kHz stereo and runs HTDemucs over it in overlapping chunks.
4. The six stems are encoded to MP3 in parallel and zipped.
5. The frontend downloads the ZIP from `GET /api/jobs/{id}/result` and loads the stems into the mixer.

## Project layout

```text
riff-apart/
|- backend/
|  |- app/
|  |  |- main.py        # FastAPI app, CORS, model warm-up on startup
|  |  |- routes.py      # separation jobs: upload, progress, result, cancel
|  |  \- separator.py   # HTDemucs loading, decoding, separation, MP3 encoding
|  |- Dockerfile        # Hugging Face Spaces image
|  \- requirements.txt
\- frontend/            # React + Vite app
```
