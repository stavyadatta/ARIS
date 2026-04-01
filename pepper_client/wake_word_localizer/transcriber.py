import json
import time
import logging
from vosk import Model, KaldiRecognizer

logger = logging.getLogger(__name__)


class TranscriberThread:
    """Consumes audio chunks from SharedState, feeds them to Vosk
    KaldiRecognizer, and checks partial/final results for wake words.

    Partial results give ~200-300ms detection latency.
    Final results serve as a backup if the partial missed it.
    """

    def __init__(self, shared_state, wake_word_detector, model_path, sample_rate=16000):
        self.shared_state = shared_state
        self.detector = wake_word_detector
        self.sample_rate = sample_rate
        self.model_path = model_path
        self.model = None
        self.recognizer = None

    def _init_model(self):
        logger.info("Loading Vosk model from %s ...", self.model_path)
        self.model = Model(self.model_path)
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)
        logger.info("Vosk model loaded successfully")

    def _reset_recognizer(self):
        """Reset after a wake word detection to avoid re-triggering."""
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)

    def run(self):
        """Main loop - intended for Thread(target=transcriber.run)."""
        self._init_model()
        logger.info("Transcriber thread started")

        while self.shared_state.running:
            chunk = self.shared_state.pop_audio()
            if chunk is None:
                time.sleep(0.05)
                continue

            is_final = self.recognizer.AcceptWaveform(chunk)

            if is_final:
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "")
                if text:
                    logger.info("Final: %s", text)
                    if self.detector.check(text):
                        logger.info(">>> WAKE WORD DETECTED in final: '%s'", text)
                        self.shared_state.signal_wake_word()
            else:
                partial = json.loads(self.recognizer.PartialResult())
                text = partial.get("partial", "")
                if text:
                    if self.detector.check(text):
                        logger.info(">>> WAKE WORD DETECTED in partial: '%s'", text)
                        self.shared_state.signal_wake_word()
                        self._reset_recognizer()

        logger.info("Transcriber thread stopped")
