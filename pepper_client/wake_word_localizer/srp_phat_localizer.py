import numpy as np
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pepper microphone positions (x, y) in metres
# Reference frame: head chain end-transform  (x = forward, y = left)
# All four mics sit at the same height (z = 0.2066 m) so we only need 2-D.
#
# Channel order when ALAudioDevice delivers ALLCHANNELS deinterleaved:
#   Ch 0 : Front-Left   Ch 1 : Front-Right
#   Ch 2 : Rear-Left    Ch 3 : Rear-Right
# ---------------------------------------------------------------------------
PEPPER_MIC_POSITIONS = np.array([
    [ 0.0313,  0.0343],   # Ch 0  Front-Left
    [ 0.0313, -0.0343],   # Ch 1  Front-Right
    [-0.0267,  0.0343],   # Ch 2  Rear-Left
    [-0.0267, -0.0343],   # Ch 3  Rear-Right
])

SPEED_OF_SOUND = 343.0   # m/s


class SRPPHATLocalizer:
    """Steered Response Power – Phase Transform (SRP-PHAT) for 2-D
    azimuth estimation on Pepper's 4-microphone array.

    How it works
    ------------
    1.  **GCC-PHAT per mic pair**
        For every pair of microphones (6 pairs from 4 mics) we compute
        the Generalised Cross-Correlation with Phase Transform.
        This gives a correlation function R(tau) that peaks at the true
        time-delay-of-arrival (TDOA) between the two mics.

        GCC-PHAT steps:
          a. FFT both signals               -> S1(f), S2(f)
          b. Cross-spectrum                  -> X(f) = S1(f) * conj(S2(f))
          c. PHAT weighting (phase-only)     -> X(f) / |X(f)|
          d. IFFT                            -> R(tau)

    2.  **Spatial grid search**
        We define a grid of candidate azimuth angles (e.g. -180 to +179
        degrees in 1-degree steps).  For each candidate angle theta and
        each mic pair (i, j) we precompute the *expected* TDOA:

            tau_ij(theta) = [ (pos_i - pos_j) . u(theta) ] / c

        where u(theta) = [cos(theta), sin(theta)] is the look-direction
        unit vector and c is the speed of sound.

    3.  **Power summation**
        For each candidate angle we look up R_ij at tau_ij(theta) for
        every pair and sum.  We also average across short overlapping
        time-frames for robustness:

            P(theta) = (1/F) * sum_frames sum_pairs  R_ij( tau_ij(theta) )

    4.  **Peak picking**
        The angle with the highest P(theta) is the estimated DOA.
        A "confidence" score = peak / mean tells us how distinct the
        peak is (higher = more reliable).

    Angle convention (matches Pepper / ALSoundLocalization):
        0      = directly in front
        +ve    = left
        -ve    = right
        +/- pi = behind
    """

    def __init__(self, mic_positions=None, sample_rate=48000,
                 nfft=1024, c=SPEED_OF_SOUND, angle_step_deg=1):
        if mic_positions is None:
            mic_positions = PEPPER_MIC_POSITIONS
        self.mic_positions = mic_positions
        self.fs = sample_rate
        self.nfft = nfft
        self.c = c
        self.n_mics = mic_positions.shape[0]

        # Candidate azimuth grid
        self.angles = np.deg2rad(np.arange(-180, 180, angle_step_deg))
        self.n_angles = len(self.angles)

        # All unique microphone pairs  (4 mics -> 6 pairs)
        self.mic_pairs = []
        for i in range(self.n_mics):
            for j in range(i + 1, self.n_mics):
                self.mic_pairs.append((i, j))
        self.n_pairs = len(self.mic_pairs)

        # Precompute GCC lookup indices for every (angle, pair)
        self._precompute_steering_indices()

        logger.info(
            "SRP-PHAT ready: %d mics, %d pairs, %d angles, nfft=%d, fs=%d",
            self.n_mics, self.n_pairs, self.n_angles, self.nfft, self.fs,
        )

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------
    def _precompute_steering_indices(self):
        """For each (angle, pair) compute the GCC index that corresponds
        to the expected TDOA.  Stored as int32 for fast numpy fancy-indexing.
        """
        cos_a = np.cos(self.angles)          # (n_angles,)
        sin_a = np.sin(self.angles)          # (n_angles,)

        # shape (n_angles, n_pairs)
        tau_idx = np.zeros((self.n_angles, self.n_pairs), dtype=np.int32)

        for p, (i, j) in enumerate(self.mic_pairs):
            dx = self.mic_positions[i, 0] - self.mic_positions[j, 0]
            dy = self.mic_positions[i, 1] - self.mic_positions[j, 1]

            # TDOA in seconds for every candidate angle
            tdoa_sec = (dx * cos_a + dy * sin_a) / self.c

            # Convert to samples, round, wrap into [0, nfft)
            tdoa_samp = np.round(tdoa_sec * self.fs).astype(np.int32)
            tau_idx[:, p] = tdoa_samp % self.nfft

        self.tau_indices = tau_idx   # (n_angles, n_pairs)

    # ------------------------------------------------------------------
    # GCC-PHAT
    # ------------------------------------------------------------------
    def _gcc_phat(self, sig1, sig2):
        """GCC-PHAT between two real signals of length nfft.

        Returns an array of length nfft whose index k represents a
        circular delay of k samples.
        """
        S1 = np.fft.rfft(sig1, n=self.nfft)
        S2 = np.fft.rfft(sig2, n=self.nfft)

        cross = S1 * np.conj(S2)
        mag = np.abs(cross)
        mag[mag < 1e-10] = 1e-10          # guard against silence / DC
        gcc = np.fft.irfft(cross / mag, n=self.nfft)

        return gcc                        # shape (nfft,)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def locate(self, multichannel_audio):
        """Estimate direction-of-arrival via SRP-PHAT.

        Parameters
        ----------
        multichannel_audio : np.ndarray, shape (n_samples, n_mics)
            Float32 audio from all microphones, normalised to [-1, 1].

        Returns
        -------
        azimuth : float   – radians (0 = front, +ve = left)
        elevation : float – always 0.0 (2-D only)
        confidence : float – peak-to-mean ratio of SRP power map
        """
        n_samples = multichannel_audio.shape[0]
        hop = self.nfft // 2                       # 50 % overlap
        n_frames = max(1, (n_samples - self.nfft) // hop + 1)

        srp_power = np.zeros(self.n_angles)

        frames_used = 0
        for f in range(n_frames):
            start = f * hop
            end = start + self.nfft
            if end > n_samples:
                break

            frame = multichannel_audio[start:end, :]
            frames_used += 1

            for p, (i, j) in enumerate(self.mic_pairs):
                gcc = self._gcc_phat(frame[:, i], frame[:, j])
                # Vectorised: look up gcc at the precomputed index for
                # every candidate angle in one go
                srp_power += gcc[self.tau_indices[:, p]]

        if frames_used > 0:
            srp_power /= frames_used

        # Peak picking
        best_idx = np.argmax(srp_power)
        best_angle = float(self.angles[best_idx])

        mean_power = np.mean(np.abs(srp_power))
        peak_power = float(srp_power[best_idx])
        confidence = peak_power / mean_power if mean_power > 1e-10 else 0.0

        return best_angle, 0.0, confidence
