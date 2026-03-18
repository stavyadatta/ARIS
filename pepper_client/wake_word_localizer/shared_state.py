import time
import threading
from collections import deque


class SharedState:
    """Thread-safe container for data shared between audio, tracking,
    transcription, and movement threads."""

    def __init__(self):
        self._lock = threading.Lock()

        # Latest sound localization data
        self._last_azimuth = 0.0
        self._last_elevation = 0.0
        self._last_confidence = 0.0
        self._last_loc_time = 0.0

        # Audio chunk queue: audio_stream -> transcriber
        # maxlen=300 (~30s at 100ms chunks) provides backpressure
        self._audio_queue = deque(maxlen=300)

        # 4-channel audio queue: srp_audio_stream -> sound_tracker (SRP-PHAT)
        # Each item is a numpy array of shape (n_samples, 4).
        # ALAudioDevice sends ~170ms buffers at 48kHz (~5.9 buffers/sec),
        # so maxlen=30 holds ~5 seconds.
        self._srp_audio_queue = deque(maxlen=30)

        # Wake word event: transcriber -> movement executor
        self._wake_event = threading.Event()

        # System running flag
        self._running = True

    # --- Sound location ---

    def update_sound_location(self, azimuth, elevation, confidence):
        with self._lock:
            self._last_azimuth = azimuth
            self._last_elevation = elevation
            self._last_confidence = confidence
            self._last_loc_time = time.time()

    def get_sound_location(self):
        """Returns (azimuth, elevation, confidence, age_seconds) or None."""
        with self._lock:
            if self._last_loc_time == 0.0:
                return None
            age = time.time() - self._last_loc_time
            return (self._last_azimuth, self._last_elevation,
                    self._last_confidence, age)

    # --- Audio queue ---

    def push_audio(self, chunk_bytes):
        self._audio_queue.append(chunk_bytes)

    def pop_audio(self):
        """Non-blocking pop. Returns None if empty."""
        try:
            return self._audio_queue.popleft()
        except IndexError:
            return None

    # --- SRP 4-channel audio queue ---

    def push_srp_audio(self, multichannel_array):
        self._srp_audio_queue.append(multichannel_array)

    def pop_srp_audio(self):
        """Non-blocking pop. Returns None if empty."""
        try:
            return self._srp_audio_queue.popleft()
        except IndexError:
            return None

    # --- Wake word event ---

    def signal_wake_word(self):
        self._wake_event.set()

    def wait_for_wake_word(self, timeout=1.0):
        """Returns True if wake word was signaled within timeout."""
        triggered = self._wake_event.wait(timeout=timeout)
        if triggered:
            self._wake_event.clear()
        return triggered

    # --- Running flag ---

    @property
    def running(self):
        with self._lock:
            return self._running

    def stop(self):
        with self._lock:
            self._running = False
        self._wake_event.set()  # unblock any waiting thread
