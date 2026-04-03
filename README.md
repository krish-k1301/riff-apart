# Riff Apart

**built by a guitarist who got tired of bad stems**

A full-stack music source separation app powered by Meta's HTDemucs. Upload any song, get 6 isolated stems i.e. vocals, drums, bass, guitar, piano, other. All downloadable, no login required.

---

> Deployment coming soon.

---

## Stack

- **Backend**: Python · FastAPI · PyTorch · Demucs (HTDemucs 6s) · librosa · soundfile
- **Frontend**: React · Vite
- **Model**: Meta's HTDemucs 6-stem — trained on a large curated dataset, hybrid transformer architecture

---

## Features

- 6-stem separation: vocals, drums, bass, guitar, piano, other
- Full song, no time limits, no login
- Download any stem as WAV
- Play stems individually or together with per-stem mute/solo
- GPU auto-detection (falls back to CPU)
- INT8 quantization on CPU for faster inference

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

Then open `http://localhost:5173`.

---

## Project layout

```
music-separator/
├── backend/
│   ├── app/
│   │   ├── audio/        # loader, processor, utils
│   │   ├── models/       # U-Net, classifier
│   │   ├── pipeline/     # Demucs pipeline, inference
│   │   └── api/          # FastAPI routes
│   ├── checkpoints/      # saved model weights
│   ├── outputs/          # separated audio
│   └── requirements.txt
└── frontend/             # React + Vite app
```
