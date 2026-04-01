import torch
import torchaudio.transforms as T


class AudioProcessor:
    """
    Handles STFT / iSTFT and spectrogram utilities using PyTorch tensors.

    All methods accept and return tensors with shapes documented per-method.
    The processor is stateless — all configuration lives in __init__.
    """

    def __init__(self, n_fft: int = 2048, hop_length: int = 512, win_length: int = 2048):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)

    # ------------------------------------------------------------------
    # STFT / iSTFT
    # ------------------------------------------------------------------

    def compute_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute the Short-Time Fourier Transform.

        Args:
            waveform: (channels, samples) or (samples,)

        Returns:
            Complex spectrogram of shape (channels, freq_bins, time_frames)
            where freq_bins = n_fft // 2 + 1.
        """
        squeezed = waveform.ndim == 1
        if squeezed:
            waveform = waveform.unsqueeze(0)

        window = self.window.to(waveform.device)
        results = []
        for ch in range(waveform.shape[0]):
            stft = torch.stft(
                waveform[ch],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                return_complex=True,
            )
            results.append(stft)

        out = torch.stack(results, dim=0)  # (channels, freq_bins, time_frames)
        return out.squeeze(0) if squeezed else out

    def compute_magnitude(self, stft_result: torch.Tensor) -> torch.Tensor:
        """
        Extract magnitude spectrogram from a complex STFT tensor.

        Args:
            stft_result: Complex tensor (..., freq_bins, time_frames)

        Returns:
            Real magnitude tensor of the same leading shape.
        """
        return stft_result.abs()

    def compute_phase(self, stft_result: torch.Tensor) -> torch.Tensor:
        """
        Extract phase spectrogram from a complex STFT tensor.

        Args:
            stft_result: Complex tensor (..., freq_bins, time_frames)

        Returns:
            Real phase tensor (angle in radians) of the same leading shape.
        """
        return stft_result.angle()

    def compute_mel_spectrogram(self, waveform: torch.Tensor, n_mels: int = 128, sr: int = 44100) -> torch.Tensor:
        """
        Compute a Mel-scale spectrogram.

        Args:
            waveform: (channels, samples) or (samples,)
            n_mels: Number of Mel filter banks.
            sr: Sample rate of the waveform.

        Returns:
            Mel spectrogram of shape (channels, n_mels, time_frames).
        """
        squeezed = waveform.ndim == 1
        if squeezed:
            waveform = waveform.unsqueeze(0)

        transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=n_mels,
        ).to(waveform.device)

        out = transform(waveform)  # (channels, n_mels, time_frames)
        return out.squeeze(0) if squeezed else out

    def inverse_stft(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a waveform from magnitude and phase spectrograms.

        Args:
            magnitude: Real tensor (channels, freq_bins, time_frames) or (freq_bins, time_frames)
            phase: Real tensor matching magnitude shape (angles in radians)

        Returns:
            Reconstructed waveform (channels, samples) or (samples,)
        """
        squeezed = magnitude.ndim == 2
        if squeezed:
            magnitude = magnitude.unsqueeze(0)
            phase = phase.unsqueeze(0)

        # Re-combine into complex spectrogram: M * e^(i*phi)
        complex_spec = magnitude * torch.exp(1j * phase)

        window = self.window.to(magnitude.device)
        results = []
        for ch in range(complex_spec.shape[0]):
            wav = torch.istft(
                complex_spec[ch],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                return_complex=False,
            )
            results.append(wav)

        out = torch.stack(results, dim=0)  # (channels, samples)
        return out.squeeze(0) if squeezed else out

    def spectrogram_to_db(self, spectrogram: torch.Tensor, ref: float = 1.0, amin: float = 1e-10) -> torch.Tensor:
        """
        Convert a power or magnitude spectrogram to decibel scale.

        Args:
            spectrogram: Non-negative tensor of any shape.
            ref: Reference value for 0 dB.
            amin: Minimum value to avoid log(0).

        Returns:
            dB-scaled tensor of the same shape.
        """
        transform = T.AmplitudeToDB(stype="magnitude", top_db=80.0)
        return transform(spectrogram)
