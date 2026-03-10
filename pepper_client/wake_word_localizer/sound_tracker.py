import time
import logging

logger = logging.getLogger(__name__)


class SoundTracker:
    """Consumes 4-channel audio from SharedState and runs SRP-PHAT to
    estimate the direction of arrival, storing the result back in
    SharedState for the MovementExecutor.

    Replaces the previous ALSoundLocalization polling approach with a
    custom SRP-PHAT implementation that processes raw microphone data.

    Data flow
    ---------
    SRPAudioService  -->  SharedState.srp_audio queue
                              |
                     SoundTracker.run() pops each buffer
                              |
                     SRPPHATLocalizer.locate(buffer)
                              |
                     SharedState.update_sound_location(az, el, conf)
    """

    def __init__(self, shared_state, srp_localizer, min_confidence=1.5):
        self.shared_state = shared_state
        self.localizer = srp_localizer
        self.min_confidence = min_confidence

    def run(self):
        """Main loop — intended for Thread(target=tracker.run)."""
        logger.info("Sound tracker thread started (SRP-PHAT-HSDA)")

        while self.shared_state.running:
            # Drain the queue and only process the LATEST buffer.
            # If locate() takes longer than ~170ms, buffers pile up.
            # Processing stale audio is pointless — we only care about
            # the most recent sound direction.
            chunk = None
            skipped = 0
            while True:
                newest = self.shared_state.pop_srp_audio()
                if newest is None:
                    break
                if chunk is not None:
                    skipped += 1
                chunk = newest
            if skipped > 0:
                logger.debug("Skipped %d stale SRP buffers", skipped)

            if chunk is None:
                time.sleep(0.01)
                continue

            try:
                azimuth, elevation, confidence = self.localizer.locate(chunk)

                logger.debug(
                    "SRP-PHAT: az=%.2f° conf=%.2f",
                    azimuth * 57.2958, confidence,
                )

                if confidence >= self.min_confidence:
                    self.shared_state.update_sound_location(
                        azimuth, elevation, confidence
                    )
                    logger.info(
                        "Sound direction updated: az=%.1f° conf=%.2f",
                        azimuth * 57.2958, confidence,
                    )

            except Exception as e:
                logger.warning("SRP-PHAT error: %s", e)

        logger.info("Sound tracker thread stopped")
