import numpy as np
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pepper microphone positions (x, y) in metres — same as srp_phat_localizer
# ---------------------------------------------------------------------------
PEPPER_MIC_POSITIONS = np.array([
    [ 0.0313,  0.0343],   # Ch 0  Front-Left
    [ 0.0313, -0.0343],   # Ch 1  Front-Right
    [-0.0267,  0.0343],   # Ch 2  Rear-Left
    [-0.0267, -0.0343],   # Ch 3  Rear-Right
])

SPEED_OF_SOUND = 343.0   # m/s


class MUSICLocalizer:
    """MUSIC (MUltiple SIgnal Classification) direction-of-arrival estimator.

    Narrowband MUSIC applied across frequency bins, then averaged to produce
    a broadband spatial spectrum.  Designed for Pepper's 4-mic planar array
    doing 2-D azimuth-only estimation.

    Algorithm
    ---------
    For each frequency bin:
      1. Estimate the spatial covariance matrix R from STFT frames.
      2. Eigendecompose R.  The eigenvectors corresponding to the smallest
         eigenvalues span the noise subspace E_n.
      3. For each candidate angle theta, compute the steering vector a(theta).
      4. MUSIC pseudo-spectrum:  P(theta) = 1 / (a^H E_n E_n^H a)

    The broadband spectrum is the average of P(theta) across frequency bins.
    The peak of the averaged spectrum gives the estimated DOA.

    Angle convention (matches Pepper / ALSoundLocalization / SRP-PHAT):
        0      = directly in front
        +ve    = left
        -ve    = right
        +/-pi  = behind
    """

    def __init__(self, mic_positions=None, sample_rate=48000,
                 nfft=1024, c=SPEED_OF_SOUND,
                 angle_step_deg=1, num_sources=1,
                 freq_range=(300, 4000), max_frames=4):
        """
        Parameters
        ----------
        mic_positions : (N, 2) array or None
            2-D mic coordinates in metres.  Defaults to Pepper's 4 mics.
        sample_rate : int
            Audio sample rate (48000 for Pepper 4-ch mode).
        nfft : int
            FFT window size in samples.
        c : float
            Speed of sound in m/s.
        angle_step_deg : int
            Angular resolution of the scanning grid (degrees).
        num_sources : int
            Number of active sound sources to assume.  For single-speaker
            scenarios, use 1.  This determines the signal/noise subspace
            split (noise subspace = n_mics - num_sources eigenvectors).
        freq_range : (int, int)
            Min and max frequencies (Hz) to include in the broadband average.
            Restricting to speech range (300-4000 Hz) improves robustness.
        max_frames : int
            Maximum STFT frames to use (limits CPU on slow hardware).
        """
        if mic_positions is None:
            mic_positions = PEPPER_MIC_POSITIONS
        self.mic_positions = mic_positions
        self.fs = sample_rate
        self.nfft = nfft
        self.c = c
        self.n_mics = mic_positions.shape[0]
        self.num_sources = num_sources
        self.max_frames = max_frames

        # Scanning grid: full 360 degrees
        self.angles = np.deg2rad(np.arange(-180, 180, angle_step_deg))
        self.n_angles = len(self.angles)

        # Frequency bin selection
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
        self.freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        self.freqs = freqs[self.freq_mask]
        self.freq_indices = np.where(self.freq_mask)[0]

        # Precompute steering vectors for all angles and selected frequencies
        # Shape: (n_freq, n_angles, n_mics) complex
        self.steering_vectors = self._precompute_steering_vectors()

    def _precompute_steering_vectors(self):
        """Precompute steering vectors a(theta, f) for all angles and frequencies."""
        # Unit direction vectors for each angle (2-D)
        # Convention: 0 = forward (+x), +ve = left (+y)
        dx = np.cos(self.angles)  # (n_angles,)
        dy = np.sin(self.angles)  # (n_angles,)
        directions = np.stack([dx, dy], axis=-1)  # (n_angles, 2)

        # Time delay for each mic at each angle: tau = mic_pos . direction / c
        # Shape: (n_angles, n_mics)
        tau = directions @ self.mic_positions.T / self.c

        # Steering vectors: a_m(theta, f) = exp(-j * 2pi * f * tau_m)
        # Shape: (n_freq, n_angles, n_mics)
        steering = np.exp(
            -1j * 2 * np.pi * self.freqs[:, None, None] * tau[None, :, :]
        )
        return steering

    def locate(self, multichannel_audio):
        """Estimate direction-of-arrival via narrowband MUSIC.

        Parameters
        ----------
        multichannel_audio : np.ndarray, shape (n_samples, n_mics)
            Float32 audio from all microphones, normalised to [-1, 1].

        Returns
        -------
        azimuth   : float  -- radians  (0 = front, +ve = left)
        elevation : float  -- always 0.0  (2-D only)
        confidence: float  -- peak-to-mean ratio of MUSIC spectrum
        """
        n_samples = multichannel_audio.shape[0]
        hop = self.nfft // 2

        # Compute STFT for each channel
        n_frames_total = max(1, (n_samples - self.nfft) // hop + 1)
        n_frames = min(n_frames_total, self.max_frames)

        # Evenly space selected frames
        if n_frames_total > self.max_frames:
            frame_step = n_frames_total / self.max_frames
            frame_indices = [int(i * frame_step) for i in range(self.max_frames)]
        else:
            frame_indices = list(range(n_frames_total))

        # Window
        window = np.hanning(self.nfft)

        # Compute STFT frames: shape (n_frames, n_mics, n_rfft)
        stft_frames = []
        for f_idx in frame_indices:
            start = f_idx * hop
            end = start + self.nfft
            if end > n_samples:
                break
            frame = multichannel_audio[start:end, :] * window[:, None]
            X = np.fft.rfft(frame, axis=0)  # (n_rfft, n_mics)
            stft_frames.append(X)

        if not stft_frames:
            return 0.0, 0.0, 0.0

        n_used = len(stft_frames)

        # Broadband MUSIC spectrum: average across selected frequency bins
        music_spectrum = np.zeros(self.n_angles)

        for k, f_bin in enumerate(self.freq_indices):
            # Estimate spatial covariance matrix for this frequency bin
            # R = (1/L) * sum(X * X^H) where X is (n_mics, 1)
            R = np.zeros((self.n_mics, self.n_mics), dtype=np.complex128)
            for stft in stft_frames:
                x = stft[f_bin, :]  # (n_mics,)
                R += np.outer(x, np.conj(x))
            R /= n_used

            # Diagonal loading for numerical stability
            R += 1e-6 * np.eye(self.n_mics)

            # Eigendecomposition (eigenvalues in ascending order)
            eigenvalues, eigenvectors = np.linalg.eigh(R)

            # Noise subspace: smallest (n_mics - num_sources) eigenvectors
            n_noise = self.n_mics - self.num_sources
            En = eigenvectors[:, :n_noise]  # (n_mics, n_noise)

            # MUSIC pseudo-spectrum for this frequency bin
            # P(theta) = 1 / (a^H * En * En^H * a)
            a = self.steering_vectors[k, :, :]  # (n_angles, n_mics)

            # a^H * En: (n_angles, n_noise)
            aH_En = np.conj(a) @ En

            # denominator: sum of |a^H * e_i|^2 for each noise eigenvector
            denom = np.sum(np.abs(aH_En) ** 2, axis=-1)  # (n_angles,)

            # Avoid division by zero
            denom = np.maximum(denom, 1e-12)

            music_spectrum += 1.0 / denom

        # Average across frequency bins
        n_freq_bins = len(self.freq_indices)
        if n_freq_bins > 0:
            music_spectrum /= n_freq_bins

        # Find peak
        peak_idx = np.argmax(music_spectrum)
        azimuth = float(self.angles[peak_idx])
        peak_val = music_spectrum[peak_idx]
        mean_val = np.mean(music_spectrum)

        # Confidence: peak-to-mean ratio
        confidence = float(peak_val / mean_val) if mean_val > 0 else 0.0

        return azimuth, 0.0, confidence
