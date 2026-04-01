import time
import logging

logger = logging.getLogger(__name__)


class MovementExecutor:
    """Waits for wake word events, reads the latest sound direction
    from SharedState, and moves the robot toward the sound source.

    Movement patterns from: sound_system/sound_localisation.py
    - Head: angleInterpolationWithSpeed("HeadYaw", azimuth, speed)
    - Body: moveTo(0, 0, azimuth) to rotate in place

    Stays facing the speaker after movement (no reset to center).
    """

    def __init__(self, session, shared_state, mode="head_and_body",
                 loc_expiry=3.0, cooldown=5.0, head_speed=0.15):
        self.motion = session.service("ALMotion")
        self.shared_state = shared_state
        self.mode = mode
        self.loc_expiry = loc_expiry
        self.cooldown = cooldown
        self.head_speed = head_speed
        self.last_movement_time = 0.0

    def run(self):
        """Main loop - waits for wake word events, then moves.
        Intended for Thread(target=executor.run)."""
        logger.info("Movement executor thread started (mode=%s)", self.mode)

        while self.shared_state.running:
            triggered = self.shared_state.wait_for_wake_word(timeout=1.0)
            if not triggered:
                continue

            if not self.shared_state.running:
                break

            # Cooldown check
            now = time.time()
            if now - self.last_movement_time < self.cooldown:
                logger.info("Movement cooldown active, skipping")
                continue

            # Get latest sound location
            loc = self.shared_state.get_sound_location()
            if loc is None:
                logger.warning("Wake word detected but no sound location available")
                continue

            azimuth, elevation, confidence, age = loc
            if age > self.loc_expiry:
                logger.warning(
                    "Sound location too old (%.1fs), skipping movement", age
                )
                continue

            logger.info(
                "Moving toward sound: az=%.2f, conf=%.2f, age=%.1fs",
                azimuth, confidence, age
            )

            self._execute_movement(azimuth)
            self.last_movement_time = time.time()

    def _execute_movement(self, azimuth):
        try:
            if self.mode in ("head", "head_and_body"):
                self.motion.setStiffnesses("Head", 1.0)
                self.motion.angleInterpolationWithSpeed(
                    "HeadYaw", azimuth, self.head_speed
                )

            if self.mode in ("body", "head_and_body"):
                self.motion.setStiffnesses("Body", 1.0)
                self.motion.moveTo(0, 0, azimuth)

        except Exception as e:
            logger.error("Movement error: %s", e)
