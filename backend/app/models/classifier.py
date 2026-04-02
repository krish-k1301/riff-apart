import torch
import torch.nn as nn
from app.data.irmas_dataset import INSTRUMENTS, INSTRUMENT_NAMES, NUM_CLASSES


class InstrumentClassifier(nn.Module):
    """
    CNN classifier for multi-label instrument detection using IRMAS classes.

    Input:  mel spectrogram  [B, 1, n_mels, time_frames]
    Output: logits           [B, 11]  (one per IRMAS instrument)

    Use BCEWithLogitsLoss during training (multi-label).
    Use predict_proba() at inference time to get per-instrument probabilities.
    """

    def __init__(self, base_channels: int = 32, num_classes: int = NUM_CLASSES):
        super().__init__()

        c = base_channels

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(c, c * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(c * 2, c * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(c * 4, c * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 8 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-stem probabilities in [0, 1] via sigmoid."""
        return torch.sigmoid(self.forward(x))
