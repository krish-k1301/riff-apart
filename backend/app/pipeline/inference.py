import torch
import torch.nn.functional as F

from app.audio.loader import load_audio, save_audio
from app.audio.processor import AudioProcessor
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

        Args:
            input_path:  Path to the source audio file.
            output_path: Destination WAV path for the separated stem.
        """
        waveform, _ = load_audio(input_path, sr=self.sr, mono=True)  # [1, total_samples]
        total_samples = waveform.shape[-1]

        output_chunks = []
        for start in range(0, total_samples, self.chunk_samples):
            chunk = waveform[:, start : start + self.chunk_samples]

            # Zero-pad the last chunk if shorter than chunk_samples
            if chunk.shape[-1] < self.chunk_samples:
                chunk = F.pad(chunk, (0, self.chunk_samples - chunk.shape[-1]))

            stft = self.processor.compute_stft(chunk)
            mix_mag = self.processor.compute_magnitude(stft)   # [1, F, T]
            mix_phase = self.processor.compute_phase(stft)     # [1, F, T]

            separated = self.separator.separate(mix_mag, mix_phase)  # [1, samples]
            output_chunks.append(separated)

        full_output = torch.cat(output_chunks, dim=-1)[:, :total_samples]
        save_audio(full_output, output_path, sr=self.sr)
