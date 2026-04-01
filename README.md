# music-separator

A music source separation web application using deep learning.

## Stack
- **Backend**: Python · FastAPI · PyTorch · torchaudio · librosa
- **Frontend**: React (Phase 4)

## Phase status
| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Audio Processing Module (STFT, loader, utils) | ✅ Complete |
| 2 | U-Net Separator + Training | Pending |
| 3 | Inference Pipeline + API | Pending |
| 4 | React Frontend | Pending |

## Quick start (backend)
```bash
cd backend
pip install -r requirements.txt
python test_phase1.py
```

## Project layout
```
music-separator/
├── backend/
│   ├── app/
│   │   ├── audio/        # loader, processor, utils  ← Phase 1
│   │   ├── models/       # U-Net, classifier          ← Phase 2
│   │   ├── pipeline/     # separator, inference       ← Phase 3
│   │   └── api/          # FastAPI routes             ← Phase 3
│   ├── data/raw/         # MUSDB18 dataset
│   ├── data/processed/   # preprocessed spectrograms
│   ├── checkpoints/      # saved model weights
│   ├── outputs/          # separated audio + plots
│   ├── requirements.txt
│   └── train.py
└── frontend/             # React app  ← Phase 4
```
