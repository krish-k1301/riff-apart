import torch

from app.models.unet import UNet
from app.audio.processor import AudioProcessor


class Separator:
    """
    Wraps a trained U-Net to separate a single target stem from a mix.

    Usage:
        sep = Separator("checkpoints/unet_vocals_best.pt")
        waveform = sep.separate(mix_mag, mix_phase)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
    ):
        self.device = torch.device(device)
        self.processor = AudioProcessor(n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        self.model = UNet()
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        # Support both a raw state_dict and a full checkpoint dict
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def separate(self, mix_mag: torch.Tensor, mix_phase: torch.Tensor) -> torch.Tensor:
        """
        Estimate the target stem waveform via mask-based separation.

        Args:
            mix_mag:   magnitude spectrogram — [1, F, T] or [B, 1, F, T]
            mix_phase: phase spectrogram     — same shape as mix_mag

        Returns:
            waveform: [channels, samples]
        """
        batched = mix_mag.ndim == 4
        if not batched:
            mix_mag = mix_mag.unsqueeze(0)
            mix_phase = mix_phase.unsqueeze(0)

        mix_mag = mix_mag.to(self.device)
        mix_phase = mix_phase.to(self.device)

        mask = self.model(mix_mag)          # [B, 1, F, T]  values in [0, 1]
        target_mag = mask * mix_mag         # estimated target magnitude

        if not batched:
            target_mag = target_mag.squeeze(0)   # [1, F, T]
            mix_phase = mix_phase.squeeze(0)

        return self.processor.inverse_stft(target_mag.cpu(), mix_phase.cpu())
