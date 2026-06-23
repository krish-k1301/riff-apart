# Riff Apart

**built by a guitarist who got tired of bad stems**

[Live site](https://riff-apart.vercel.app/)

A full-stack music source separation app powered by Meta's HTDemucs. Upload any song, get 6 isolated stems i.e. vocals, drums, bass, guitar, piano, other. All downloadable, no login required.

## How to use

1. Open [riff-apart.vercel.app](https://riff-apart.vercel.app/).
2. Upload a song from your device.
3. Wait while Riff Apart separates it into vocals, drums, bass, guitar, piano, and other.
4. Play stems individually or together in the mixer.
5. Use mute/solo controls to isolate parts.
6. Download any stem as a WAV file.

## Stack

- **Backend**: Python · FastAPI · PyTorch · Demucs (HTDemucs 6s) · librosa · soundfile
- **Frontend**: React · Vite
- **Model**: Meta's HTDemucs 6-stem - trained on a large curated dataset, hybrid transformer architecture

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
