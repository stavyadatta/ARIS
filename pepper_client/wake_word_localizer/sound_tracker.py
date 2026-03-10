import time
import logging

logger = logging.getLogger(__name__)


class SoundTracker:
    """Polls ALSoundLocalization/SoundLocated from ALMemory and stores
    the latest direction in SharedState.

    Pattern from: sound_system/sound_localisation.py
    - subscribe via sound_localization.subscribe("SoundLocated")
    - read via memory.getData("ALSoundLocalization/SoundLocated")
    - data format: [timestamp, [azimuth, elevation, confidence], energy]
    """

    def __init__(self, session, shared_state, poll_interval=0.1,
                 min_confidence=0.0, sensitivity=0.8):
        self.memory = session.service("ALMemory")
        self.sound_localization = session.service("ALSoundLocalization")
        self.shared_state = shared_state
        self.poll_interval = poll_interval
        self.min_confidence = min_confidence
        self.sensitivity = sensitivity
        # Track the last raw data to avoid re-processing identical readings
        self._last_raw_data = None

    def run(self):
        """Main polling loop - intended for Thread(target=tracker.run)."""
        logger.info("Sound tracker thread started")
        self.sound_localization.subscribe("SoundLocated")
        self.sound_localization.setParameter("Sensibility", self.sensitivity)
        logger.info("Subscribed to ALSoundLocalization (sensitivity=%.1f)", self.sensitivity)

        try:
            while self.shared_state.running:
                try:
                    data = self.memory.getData("ALSoundLocalization/SoundLocated")

                    if not data or len(data) < 2:
                        time.sleep(self.poll_interval)
                        continue

                    # Skip if this is the exact same reading as last time
                    # if data == self._last_raw_data:
                    #     time.sleep(self.poll_interval)
                    #     continue
                    # self._last_raw_data = data
                    # logger.info("The program is entering ALSoundLocalization thread")

                    # Parse: data = [timestamp_or_id, [azimuth, elevation, confidence], ...]
                    sound_info = data[1]
                    if not sound_info or len(sound_info) < 3:
                        logger.warning("Unexpected sound data format: %s", data)
                        time.sleep(self.poll_interval)
                        continue

                    azimuth = sound_info[0]
                    elevation = sound_info[1]
                    confidence = sound_info[2]

                    # logger.info(
                    #     "The Sound location data is az=%.2f, el=%.2f and conf=%.2f",
                    #     azimuth, elevation, confidence
                    #
                    # )

                    if confidence >= self.min_confidence:
                        self.shared_state.update_sound_location(
                            azimuth, elevation, confidence
                        )
                        # logger.info(
                        #     "New sound location: az=%.2f el=%.2f conf=%.2f",
                        #     azimuth, elevation, confidence
                        # )
                    else:
                        logger.debug(
                            "Sound below confidence threshold: conf=%.2f < %.2f",
                            confidence, self.min_confidence
                        )

                except Exception as e:
                    logger.warning("Sound tracker poll error: %s", e)

                time.sleep(self.poll_interval)
        finally:
            try:
                self.sound_localization.unsubscribe("SoundLocated")
            except Exception:
                pass
            logger.info("Sound tracker thread stopped")
