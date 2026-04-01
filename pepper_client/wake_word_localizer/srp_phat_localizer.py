import numpy as np
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pepper microphone positions (x, y) in metres
# Reference frame: head chain end-transform  (x = forward, y = left)
# All four mics sit at the same height (z ≈ 0.2066 m) so we do 2-D only.
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
    """SRP-PHAT-HSDA: Steered Response Power – Phase Transform with
    Hierarchical Search, Directivity model, and Automatic calibration.

    Based on Grondin & Michaud (2019) "Lightweight and Optimized Sound
    Source Localization and Tracking Methods for Open and Closed
    Microphone Array Configurations", adapted for Pepper's 4-mic
    array in 2-D (azimuth only).

    Three key improvements over plain SRP-PHAT
    -------------------------------------------

    1.  **Hierarchical Search (HS)**
        Instead of scanning all 360 angles at fine resolution every
        time, we use a two-level grid:

          Level 1 – COARSE grid (e.g. every 10°  → 36 points)
            Scan all coarse directions.  Find the best one.

          Level 2 – FINE grid (e.g. every 1°)
            Only scan a small window (±refine_radius) around the
            best coarse direction.

        This cuts the number of GCC lookups by ~5-10× while keeping
        the same 1° output resolution.

    2.  **Microphone Directivity model (D)**
        Each mic has a forward-facing direction (auto-computed as the
        outward normal from the array centre) and an acceptance
        half-angle.  For any candidate scan direction θ, a mic pair
        (i, j) is only used if BOTH mics can "see" that direction:

            cos(angle between mic_dir and scan_dir)  ≥  cos(acceptance)

        This avoids wasting computation on mic pairs where one mic is
        acoustically shadowed by the robot's head, and reduces false
        detections for closed microphone arrays.

    3.  **Maximum Sliding Window – Automatic calibration (A)**
        Standard SRP-PHAT looks up a single GCC-PHAT value at the
        exact expected TDOA sample index.  But real TDOA has
        uncertainty from:
          - finite sampling (fractional-sample true delay)
          - mic position tolerances
          - head diffraction bending the wavefront

        MSW replaces the single-sample lookup with the maximum
        GCC-PHAT value within a ±delta window around the expected
        TDOA index:

            R_msw(τ) = max_{d ∈ [-δ, +δ]}  R(τ + d)

        The delta is auto-computed from the mic pair distance and
        sample rate so that it covers the positioning uncertainty.

    Algorithm flow (per audio buffer)
    ---------------------------------
    1. Compute GCC-PHAT for each mic pair, across overlapping frames
    2. Apply MSW filter to each GCC (replaces each sample with the
       max in a ±delta window)
    3. COARSE scan: for each coarse angle, sum the MSW-filtered GCC
       values at the expected TDOAs, but only for directivity-valid
       mic pairs.  Find the best coarse direction.
    4. FINE scan: for each fine angle within ±refine_radius of the
       best coarse direction, same summation.
    5. Peak pick on the fine grid → estimated azimuth.
    6. Confidence = peak / mean of the fine power map.

    Angle convention (matches Pepper / ALSoundLocalization):
        0      = directly in front
        +ve    = left
        -ve    = right
        ±π     = behind
    """

    def __init__(self, mic_positions=None, sample_rate=48000,
                 nfft=1024, c=SPEED_OF_SOUND,
                 coarse_step_deg=10, fine_step_deg=1,
                 refine_radius_deg=15,
                 mic_acceptance_deg=150, msw_delta=None,
                 max_frames=2, enable_msw=True):
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
        coarse_step_deg : int
            Angular step for the coarse grid (degrees).
        fine_step_deg : int
            Angular step for the fine grid (degrees).
        refine_radius_deg : int
            Half-width of the fine search window around the best
            coarse direction (degrees).
        mic_acceptance_deg : float
            Each mic's acceptance half-angle for the directivity
            model (degrees).  150° is generous (only rejects sounds
            from almost directly behind the mic).
        msw_delta : int or None
            Half-width of the Maximum Sliding Window (samples).
            None = auto-compute from mic geometry.
        max_frames : int
            Maximum number of time-frames to process per audio buffer.
            Limits CPU on slow hardware (e.g. Raspberry Pi).  At 48 kHz
            with nfft=1024 a 170ms buffer has ~14 frames; processing
            all of them means 252 FFTs.  4 frames → 72 FFTs.
        """
        if mic_positions is None:
            mic_positions = PEPPER_MIC_POSITIONS
        self.mic_positions = mic_positions
        self.fs = sample_rate
        self.nfft = nfft
        self.c = c
        self.n_mics = mic_positions.shape[0]

        # ---- Performance ----
        self.max_frames = max_frames
        self.enable_msw = enable_msw

        # ---- Hierarchical grid parameters ----
        self.coarse_step = coarse_step_deg
        self.fine_step = fine_step_deg
        self.refine_radius = refine_radius_deg

        # Coarse grid: e.g. -180, -170, ..., 170  (36 points at 10° step)
        self.coarse_angles = np.deg2rad(
            np.arange(-180, 180, coarse_step_deg)
        )
        self.n_coarse = len(self.coarse_angles)

        # Fine grid: full 1° resolution (used as a lookup; only a
        # subset is scanned per frame)
        self.fine_angles = np.deg2rad(
            np.arange(-180, 180, fine_step_deg)
        )
        self.n_fine = len(self.fine_angles)

        # ---- Mic pairs ----
        self.mic_pairs = []
        for i in range(self.n_mics):
            for j in range(i + 1, self.n_mics):
                self.mic_pairs.append((i, j))
        self.n_pairs = len(self.mic_pairs)

        # ---- Directivity model ----
        self.mic_directions = self._compute_mic_directions()
        self.cos_acceptance = np.cos(np.deg2rad(mic_acceptance_deg))

        # Precompute directivity masks for coarse and fine grids
        self.coarse_dir_mask = self._build_directivity_mask(self.coarse_angles)
        self.fine_dir_mask = self._build_directivity_mask(self.fine_angles)

        # ---- Steering indices ----
        self.coarse_tau = self._build_tau_indices(self.coarse_angles)
        self.fine_tau = self._build_tau_indices(self.fine_angles)

        # ---- MSW delta (auto or manual) ----
        if msw_delta is None:
            self.msw_delta = self._auto_msw_delta()
        else:
            self.msw_delta = int(msw_delta)

        logger.info(
            "SRP-PHAT-HSDA ready: %d mics, %d pairs, "
            "coarse=%d pts (%d°), fine=%d pts (%d°), "
            "refine=±%d°, msw_delta=%d, acceptance=%d°",
            self.n_mics, self.n_pairs,
            self.n_coarse, coarse_step_deg,
            self.n_fine, fine_step_deg,
            refine_radius_deg, self.msw_delta,
            mic_acceptance_deg,
        )

    # ==================================================================
    # Initialisation helpers
    # ==================================================================

    def _compute_mic_directions(self):
        """Auto-compute each mic's outward-facing direction as the
        unit vector from the array centroid to the mic position.

        For Pepper this gives sensible results:
          Front-Left  mic → points forward-left
          Front-Right mic → points forward-right
          Rear-Left   mic → points backward-left
          Rear-Right  mic → points backward-right
        """
        centre = np.mean(self.mic_positions, axis=0)
        dirs = self.mic_positions - centre
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        return dirs / norms                    # (n_mics, 2)

    def _build_directivity_mask(self, angles):
        """For each (angle, pair), True if both mics can 'see' that
        direction.

        Returns bool array of shape (n_angles, n_pairs).
        """
        n = len(angles)
        # Unit look-direction vectors  (n_angles, 2)
        look = np.column_stack([np.cos(angles), np.sin(angles)])

        mask = np.ones((n, self.n_pairs), dtype=bool)

        for p, (i, j) in enumerate(self.mic_pairs):
            # Cosine of angle between mic direction and look direction
            cos_i = look @ self.mic_directions[i]   # (n_angles,)
            cos_j = look @ self.mic_directions[j]   # (n_angles,)
            mask[:, p] = (cos_i >= self.cos_acceptance) & \
                         (cos_j >= self.cos_acceptance)

        return mask

    def _build_tau_indices(self, angles):
        """Precompute the GCC sample index for each (angle, pair).

        Returns int32 array of shape (n_angles, n_pairs).
        """
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        n = len(angles)
        tau = np.zeros((n, self.n_pairs), dtype=np.int32)

        for p, (i, j) in enumerate(self.mic_pairs):
            dx = self.mic_positions[i, 0] - self.mic_positions[j, 0]
            dy = self.mic_positions[i, 1] - self.mic_positions[j, 1]
            tdoa_sec = (dx * cos_a + dy * sin_a) / self.c
            tdoa_samp = np.round(tdoa_sec * self.fs).astype(np.int32)
            tau[:, p] = tdoa_samp % self.nfft

        return tau

    def _auto_msw_delta(self):
        """Automatically compute the MSW half-width from mic geometry.

        We set delta so the window covers ±1 sample of TDOA uncertainty
        plus a margin for head diffraction.  Practically this is
        1–3 samples for Pepper's small array at 48 kHz.
        """
        max_dist = 0.0
        for i, j in self.mic_pairs:
            d = np.linalg.norm(
                self.mic_positions[i] - self.mic_positions[j]
            )
            if d > max_dist:
                max_dist = d

        # Max TDOA in samples
        max_tdoa_samp = max_dist / self.c * self.fs

        # Delta = ceil of 20 % of max TDOA, minimum 1
        delta = max(1, int(np.ceil(max_tdoa_samp * 0.2)))
        return delta

    # ==================================================================
    # GCC-PHAT
    # ==================================================================

    def _gcc_phat(self, sig1, sig2):
        """GCC-PHAT between two real signals.

        Returns an array of length nfft whose index k represents a
        circular delay of k samples.
        """
        S1 = np.fft.rfft(sig1, n=self.nfft)
        S2 = np.fft.rfft(sig2, n=self.nfft)

        cross = S1 * np.conj(S2)
        mag = np.abs(cross)
        mag[mag < 1e-10] = 1e-10
        gcc = np.fft.irfft(cross / mag, n=self.nfft)

        return gcc                             # (nfft,)

    # ==================================================================
    # Maximum Sliding Window
    # ==================================================================

    def _apply_msw(self, gcc):
        """Replace each GCC sample with the maximum in a ±delta window.

        This accounts for TDOA uncertainty: instead of requiring the
        true delay to land exactly on one sample, we accept the
        strongest correlation within a small neighbourhood.

        Implemented via numpy.roll which wraps around circularly
        (matching the circular nature of the IFFT output).
        """
        delta = self.msw_delta
        result = gcc.copy()
        for d in range(1, delta + 1):
            result = np.maximum(result, np.roll(gcc, d))
            result = np.maximum(result, np.roll(gcc, -d))
        return result

    # ==================================================================
    # Main entry point
    # ==================================================================

    def locate(self, multichannel_audio):
        """Estimate direction-of-arrival via SRP-PHAT-HSDA.

        Parameters
        ----------
        multichannel_audio : np.ndarray, shape (n_samples, n_mics)
            Float32 audio from all microphones, normalised to [-1, 1].

        Returns
        -------
        azimuth   : float  – radians  (0 = front, +ve = left)
        elevation : float  – always 0.0  (2-D only)
        confidence: float  – peak-to-mean ratio of SRP power map
        """
        n_samples = multichannel_audio.shape[0]
        hop = self.nfft // 2
        n_frames_total = max(1, (n_samples - self.nfft) // hop + 1)

        # Cap frames to limit CPU usage (e.g. 14 frames → 4).
        # Evenly space the selected frames across the buffer so we
        # still get temporal coverage.
        n_frames = min(n_frames_total, self.max_frames)
        if n_frames_total > self.max_frames:
            frame_step = n_frames_total / self.max_frames
            frame_indices = [int(i * frame_step) for i in range(self.max_frames)]
        else:
            frame_indices = list(range(n_frames_total))

        # --- Step 1: Compute GCC-PHAT + MSW for each mic pair ----------
        # Average GCC across selected frames, then apply MSW
        gcc_all = [np.zeros(self.nfft) for _ in range(self.n_pairs)]
        frames_used = 0

        for f in frame_indices:
            start = f * hop
            end = start + self.nfft
            if end > n_samples:
                break
            frame = multichannel_audio[start:end, :]
            frames_used += 1

            for p, (i, j) in enumerate(self.mic_pairs):
                gcc_all[p] += self._gcc_phat(frame[:, i], frame[:, j])

        if frames_used > 0:
            for p in range(self.n_pairs):
                gcc_all[p] /= frames_used

        # Apply Maximum Sliding Window to each pair's GCC (optional)
        if self.enable_msw:
            gcc_msw = [self._apply_msw(g) for g in gcc_all]
        else:
            gcc_msw = gcc_all

        # --- Step 2: COARSE scan (Hierarchical level 1) -----------------
        coarse_power = np.zeros(self.n_coarse)

        for p in range(self.n_pairs):
            gcc = gcc_msw[p]
            tau = self.coarse_tau[:, p]           # (n_coarse,)
            mask = self.coarse_dir_mask[:, p]     # (n_coarse,) bool

            # Look up GCC at expected TDOA, zero out invalid directions
            values = gcc[tau]
            values[~mask] = 0.0
            coarse_power += values

        best_coarse_idx = int(np.argmax(coarse_power))
        best_coarse_angle_deg = np.rad2deg(
            self.coarse_angles[best_coarse_idx]
        )

        # --- Step 3: FINE scan (Hierarchical level 2) -------------------
        # Determine which fine-grid indices fall within ±refine_radius
        # of the best coarse angle
        fine_angles_deg = np.rad2deg(self.fine_angles)
        lo = best_coarse_angle_deg - self.refine_radius
        hi = best_coarse_angle_deg + self.refine_radius

        # Handle wrap-around (e.g. best coarse = 175°, radius = 15°)
        if lo < -180:
            fine_mask = (fine_angles_deg >= (lo + 360)) | \
                        (fine_angles_deg <= hi)
        elif hi >= 180:
            fine_mask = (fine_angles_deg >= lo) | \
                        (fine_angles_deg <= (hi - 360))
        else:
            fine_mask = (fine_angles_deg >= lo) & \
                        (fine_angles_deg <= hi)

        fine_indices = np.where(fine_mask)[0]

        if len(fine_indices) == 0:
            # Fallback: use coarse result directly
            best_angle = float(self.coarse_angles[best_coarse_idx])
            return best_angle, 0.0, float(coarse_power[best_coarse_idx])

        fine_power = np.zeros(len(fine_indices))

        for p in range(self.n_pairs):
            gcc = gcc_msw[p]
            tau_subset = self.fine_tau[fine_indices, p]
            mask_subset = self.fine_dir_mask[fine_indices, p]

            values = gcc[tau_subset]
            values[~mask_subset] = 0.0
            fine_power += values

        # --- Step 4: Peak pick on fine grid -----------------------------
        best_fine_local = int(np.argmax(fine_power))
        best_fine_global = fine_indices[best_fine_local]
        best_angle = float(self.fine_angles[best_fine_global])

        # Confidence: peak / mean of the fine search power
        mean_power = np.mean(np.abs(fine_power))
        peak_power = float(fine_power[best_fine_local])
        confidence = peak_power / mean_power if mean_power > 1e-10 else 0.0

        return best_angle, 0.0, confidence
