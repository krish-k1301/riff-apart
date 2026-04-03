# Riff Apart

**built by a guitarist who got tired of bad stems**

A full-stack music source separation app powered by Meta's HTDemucs. Upload any song, get 6 isolated stems — vocals, drums, bass, guitar, piano, other — all downloadable, no login required.

---

## The story

It started with *Beete Lamhe* by KK. The guitar intro on that track is something else , the tone, the phrasing, everything about it was stuck in my head. As a guitarist I wanted to figure out exactly what was going on in that part so I could dial in the same tone on my multiprocessor.

I found a YouTube lesson for it, but tone is a whole different problem. I needed to actually hear the isolated guitar ,no vocals, no drums, just the guitar sitting in the mix. So I went to one of those online stem separators. It worked. But it wouldn't let me download the isolated track, capped me at 1 minute without logging in, and I was too lazy to log in.

So I built Riff Apart instead.

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
