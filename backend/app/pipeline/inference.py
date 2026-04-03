import torch
import torch.nn.functional as F

from app.audio.loader import load_audio, save_audio
from app.audio.processor import AudioProcessor
from app.models.unet import UNet
from app.pipeline.separator import Separator


class InferencePipeline:
    """
    End-to-end inference: audio file path → separated stem audio file.

    Splits the input into fixed-length non-overlapping chunks, separates each
    chunk individually, then concatenates and trims to the original length.

    Usage:
        pipeline = InferencePipeline("checkpoints/unet_vocals_best.pt")
        pipeline.run("song.wav", "vocals_out.wav")
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        sr: int = 44100,
        chunk_duration: float = 5.0,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
    ):
        self.sr = sr
        self.chunk_samples = int(chunk_duration * sr)
        self.processor = AudioProcessor(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.separator = Separator(
            model_path=model_path,
            device=device,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

    def run(self, input_path: str, output_path: str) -> None:
        """
        Separate the audio at input_path and write the result to output_path.

        Uses 50% overlap-add with a Hann window to eliminate chunk-boundary
        clicks/distortion that hard concatenation produces.
        """
        waveform, _ = load_audio(input_path, sr=self.sr, mono=True)  # [1, total_samples]
        total_samples = waveform.shape[-1]

        hop = self.chunk_samples // 2  # 50% overlap
        window = torch.hann_window(self.chunk_samples)  # fade in/out per chunk

        # Accumulator and normalisation buffer for overlap-add
        out_len = total_samples + self.chunk_samples
        output_buf = torch.zeros(1, out_len)
        weight_buf = torch.zeros(out_len)

        start = 0
        while start < total_samples:
            chunk = waveform[:, start : start + self.chunk_samples]

            # Zero-pad the last chunk if shorter than chunk_samples
            pad_len = self.chunk_samples - chunk.shape[-1]
            if pad_len > 0:
                chunk = F.pad(chunk, (0, pad_len))

            stft = self.processor.compute_stft(chunk)
            mix_mag = self.processor.compute_magnitude(stft)   # [1, F, T]
            mix_phase = self.processor.compute_phase(stft)     # [1, F, T]

            separated = self.separator.separate(mix_mag, mix_phase)  # [1, samples]

            # Apply Hann window to the separated chunk before accumulating
            end = start + self.chunk_samples
            output_buf[:, start:end] += separated * window
            weight_buf[start:end] += window

            start += hop

        # Normalise by accumulated window weights to undo the overlap-add scaling
        weight_buf = weight_buf.clamp(min=1e-8)
        full_output = (output_buf / weight_buf)[:, :total_samples]
        save_audio(full_output, output_path, sr=self.sr)


class MultiStemInferencePipeline:
    """
    Separates all 4 stems simultaneously using Wiener filtering.

    Instead of each model working independently, all four masks are computed per
    chunk and normalised so they compete for each time-frequency bin:

        wiener_mask[stem] = mask[stem]^2 / sum(mask[s]^2 for all s)

    This prevents two models from both claiming the same energy and significantly
    reduces bleed between stems (especially vocals ↔ guitar, harmony layers).

    Usage:
        pipeline = MultiStemInferencePipeline(
            checkpoint_dir="checkpoints/",
            stems=["vocals", "drums", "bass", "other"],
        )
        pipeline.run("song.wav", output_dir="outputs/")
        # writes outputs/vocals.wav, outputs/drums.wav, ...
    """

    STEMS = ["vocals", "drums", "bass", "other"]

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cpu",
        sr: int = 44100,
        chunk_duration: float = 5.0,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
    ):
        from pathlib import Path

        self.sr = sr
        self.chunk_samples = int(chunk_duration * sr)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.device = torch.device(device)
        self.processor = AudioProcessor(n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        checkpoint_dir = Path(checkpoint_dir)
        self.models: dict[str, UNet] = {}
        for stem in self.STEMS:
            ckpt_path = checkpoint_dir / f"unet_{stem}_best.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
            model = UNet()
            checkpoint = torch.load(str(ckpt_path), map_location=self.device, weights_only=True)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            self.models[stem] = model

    @torch.no_grad()
    def _separate_chunk(self, mix_mag: torch.Tensor, mix_phase: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Run all 4 U-Nets on a single chunk and apply Wiener filtering.

        Returns a dict of stem → waveform [1, samples].
        """
        mix_mag = mix_mag.unsqueeze(0).to(self.device)   # [1, 1, F, T]
        mix_phase = mix_phase.unsqueeze(0).to(self.device)

        # Collect raw masks from each model
        masks = {stem: self.models[stem](mix_mag) for stem in self.STEMS}  # each [1, 1, F, T]

        # Wiener filter: normalise masks so they sum-of-squares to 1 per bin
        mask_sum_sq = sum(m ** 2 for m in masks.values()).clamp(min=1e-10)
        wiener_masks = {stem: (masks[stem] ** 2) / mask_sum_sq for stem in self.STEMS}

        results: dict[str, torch.Tensor] = {}
        for stem in self.STEMS:
            target_mag = (wiener_masks[stem] * mix_mag).squeeze(0)   # [1, F, T]
            phase = mix_phase.squeeze(0)
            results[stem] = self.processor.inverse_stft(target_mag.cpu(), phase.cpu())  # [1, samples]

        return results

    def run(self, input_path: str, output_dir: str) -> dict[str, str]:
        """
        Separate all stems from input_path and write each to output_dir/{stem}.wav.

        Returns a dict of stem → output file path.
        """
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        waveform, _ = load_audio(input_path, sr=self.sr, mono=True)  # [1, total_samples]
        total_samples = waveform.shape[-1]

        hop = self.chunk_samples // 2
        window = torch.hann_window(self.chunk_samples)

        out_len = total_samples + self.chunk_samples
        output_bufs = {stem: torch.zeros(1, out_len) for stem in self.STEMS}
        weight_buf = torch.zeros(out_len)

        start = 0
        while start < total_samples:
            chunk = waveform[:, start : start + self.chunk_samples]
            pad_len = self.chunk_samples - chunk.shape[-1]
            if pad_len > 0:
                chunk = F.pad(chunk, (0, pad_len))

            stft = self.processor.compute_stft(chunk)
            mix_mag = self.processor.compute_magnitude(stft)   # [1, F, T]
            mix_phase = self.processor.compute_phase(stft)

            separated = self._separate_chunk(mix_mag, mix_phase)

            end = start + self.chunk_samples
            for stem in self.STEMS:
                output_bufs[stem][:, start:end] += separated[stem] * window
            weight_buf[start:end] += window

            start += hop

        weight_buf = weight_buf.clamp(min=1e-8)
        output_paths: dict[str, str] = {}
        for stem in self.STEMS:
            full_output = (output_bufs[stem] / weight_buf)[:, :total_samples]
            out_path = str(output_dir / f"{stem}.wav")
            save_audio(full_output, out_path, sr=self.sr)
            output_paths[stem] = out_path

        return output_paths
