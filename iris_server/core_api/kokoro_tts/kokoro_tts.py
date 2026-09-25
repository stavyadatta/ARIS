"""Generate Iris's speech here instead of on the robot.

The G1's built-in text-to-speech is Chinese-first with English as speaker id 1,
and it sounds like it. Kokoro is a small open-weight model -- 82 million
parameters, Apache 2.0 -- that sounds markedly more human, and running it on the
workstation keeps the robot's Jetson free.

The audio travels to the robot inside the existing reply payload as base64,
rather than as a new protobuf field. That is deliberate: the client's C++ stubs
were generated with protobuf 3.6.1, this machine has 3.12.4, and regenerating
them risks a robot that will not build for a change that adds one field. The
reply already travels as JSON, so a key costs nothing.

PlayStream wants raw 16 kHz mono 16-bit samples, so that is what comes out.
"""

import base64
import io

import numpy as np

# What AudioClient::PlayStream accepts. Not a preference -- the robot's audio
# service rejects anything else.
ROBOT_SAMPLE_RATE = 16000
KOKORO_SAMPLE_RATE = 24000

# af_heart is the warmest of the American English voices. Changing this is the
# one knob worth turning if Iris should sound different.
DEFAULT_VOICE = "af_heart"
LANGUAGE_CODE = "a"   # American English


class _KokoroTts:
    """Lazily loaded, because importing Kokoro costs seconds and most runs never
    speak. Warmed up explicitly at start-up instead, like the other models."""

    def __init__(self) -> None:
        self._pipeline = None

    def _ensure_loaded(self):
        if self._pipeline is None:
            import torch
            from kokoro import KPipeline

            # Measured on an RTX 4090: 31 ms for 6.9 s of speech, 225x real
            # time. On CPU the same model manages 2x, which would put a second
            # and a half onto every reply -- so this is worth being explicit
            # about rather than letting the library guess.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                print("[kokoro] no CUDA device; speech will be ~100x slower "
                      "and will show up in turn latency")
            self._pipeline = KPipeline(lang_code=LANGUAGE_CODE, device=device)
        return self._pipeline

    def warm_up(self) -> None:
        """Pay the model-load and kernel-compilation cost before the first
        person speaks, not during. Measured: first call 2.25 s, every call
        after it 0.03 s."""
        self.speech_pcm("ready")

    def speech_pcm(self, text: str, voice: str = DEFAULT_VOICE):
        """16 kHz mono 16-bit PCM for `text`, or None if speech is unavailable.

        Returning None rather than raising is deliberate: the client falls back
        to the robot's own text-to-speech, so a failure here costs voice quality
        rather than the whole turn.
        """
        if not text or not text.strip():
            return None
        try:
            from scipy.signal import resample_poly

            pipeline = self._ensure_loaded()
            audio = np.concatenate([chunk for _, _, chunk in pipeline(text, voice=voice)])
            resampled = resample_poly(audio, ROBOT_SAMPLE_RATE, KOKORO_SAMPLE_RATE)
            clipped = np.clip(resampled, -1.0, 1.0)
            return (clipped * 32767).astype("<i2").tobytes()
        except Exception as e:
            print(f"[kokoro] speech generation failed, falling back to the "
                  f"robot's own voice: {e}")
            return None

    def speech_base64(self, text: str, voice: str = DEFAULT_VOICE):
        """The same audio, encoded for the JSON reply payload."""
        pcm = self.speech_pcm(text, voice)
        return base64.b64encode(pcm).decode("ascii") if pcm else None


KokoroTts = _KokoroTts()
