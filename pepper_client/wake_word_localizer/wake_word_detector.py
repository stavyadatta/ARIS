import re
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Detects wake word variants in transcribed text using exact set
    membership and fuzzy string matching via difflib.SequenceMatcher.

    Uses only the standard library (no extra deps on Raspberry Pi).
    """

    def __init__(self, wake_words, fuzzy_threshold=75):
        self.wake_words = {w.lower() for w in wake_words}
        self.fuzzy_threshold = fuzzy_threshold / 100.0

    def check(self, text):
        """Returns True if any wake word variant is detected in text."""
        words = re.findall(r"[a-zA-Z']+", text.lower())

        for word in words:
            # Exact match (fast path)
            if word in self.wake_words:
                logger.debug("Exact match: '%s'", word)
                return True

            # Fuzzy match against each variant
            for variant in self.wake_words:
                ratio = SequenceMatcher(None, word, variant).ratio()
                if ratio >= self.fuzzy_threshold:
                    logger.debug(
                        "Fuzzy match: '%s' ~ '%s' (ratio=%.2f)",
                        word, variant, ratio
                    )
                    return True

        return False
