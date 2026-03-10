import qi
import os
import sys
import signal
import argparse
import logging
import time
from threading import Thread

from wake_word_localizer.config import (
    PEPPER_IP, PEPPER_PORT, PI_LISTEN_IP, PI_LISTEN_PORT,
    VOSK_MODEL_PATH, WAKE_WORDS, FUZZY_THRESHOLD,
    SOUND_LOC_EXPIRY, SRP_NFFT, SRP_ANGLE_STEP, SRP_MIN_CONFIDENCE,
    MOVEMENT_COOLDOWN, MOVEMENT_MODE, HEAD_SPEED, SAMPLE_RATE,
)
from wake_word_localizer.shared_state import SharedState
from wake_word_localizer.audio_stream import WakeWordAudioService
from wake_word_localizer.srp_audio_stream import SRPAudioService
from wake_word_localizer.transcriber import TranscriberThread
from wake_word_localizer.wake_word_detector import WakeWordDetector
from wake_word_localizer.srp_phat_localizer import SRPPHATLocalizer
from wake_word_localizer.sound_tracker import SoundTracker
from wake_word_localizer.movement_executor import MovementExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("wake_word_main")


def main():
    parser = argparse.ArgumentParser(
        description="Wake-word-triggered sound localization for Pepper"
    )
    parser.add_argument("--ip", type=str, default=PEPPER_IP,
                        help="Pepper robot IP address")
    parser.add_argument("--port", type=int, default=PEPPER_PORT,
                        help="NAOqi port number")
    parser.add_argument("--model", type=str, default=VOSK_MODEL_PATH,
                        help="Path to Vosk model directory")
    parser.add_argument("--mode", type=str, default=MOVEMENT_MODE,
                        choices=["head", "body", "head_and_body"],
                        help="Movement mode on wake word detection")
    parser.add_argument("--listen-ip", type=str, default=PI_LISTEN_IP,
                        help="IP for qi session listener")
    parser.add_argument("--listen-port", type=int, default=PI_LISTEN_PORT,
                        help="Port for qi session listener")
    args = parser.parse_args()

    # Connect to Pepper (pattern from pepper_middleware/pepper.py)
    pepper_url = "tcp://{}:{}".format(args.ip, args.port)
    app = qi.Application(["WakeWordLocalizer", "--qi-url=" + pepper_url])
    app.start()
    session = app.session
    logger.info("Connected to Pepper at %s", pepper_url)

    # Wake the robot (enables motor stiffness and standing posture)
    motion = session.service("ALMotion")
    motion.wakeUp()
    logger.info("Robot is awake")

    # Shared state
    state = SharedState()

    # ---- Audio services ----

    # 1-channel 16 kHz for Vosk wake-word detection
    audio_service = WakeWordAudioService(state)
    listen_url = "tcp://{}:{}".format(args.listen_ip, args.listen_port)
    session.listen(listen_url)
    session.registerService("WakeWordAudioService", audio_service)
    audio_service.init_service(session)
    logger.info("Wake-word audio service registered, listening on %s", listen_url)

    # 4-channel 48 kHz for SRP-PHAT localization
    srp_audio = SRPAudioService(state)
    session.registerService("SRPAudioService", srp_audio)
    srp_audio.init_service(session)
    logger.info("SRP audio service registered (4-ch, 48 kHz)")

    # ---- SRP-PHAT localizer ----
    srp_localizer = SRPPHATLocalizer(
        nfft=SRP_NFFT,
        angle_step_deg=SRP_ANGLE_STEP,
    )

    # ---- Workers ----
    detector = WakeWordDetector(WAKE_WORDS, FUZZY_THRESHOLD)
    transcriber = TranscriberThread(state, detector, args.model, SAMPLE_RATE)

    tracker = SoundTracker(
        state, srp_localizer,
        min_confidence=SRP_MIN_CONFIDENCE,
    )

    executor = MovementExecutor(
        session, state,
        mode=args.mode,
        loc_expiry=SOUND_LOC_EXPIRY,
        cooldown=MOVEMENT_COOLDOWN,
        head_speed=HEAD_SPEED,
    )

    # Start threads (pattern from pepper.py lines 321-338)
    threads = []

    transcriber_thread = Thread(target=transcriber.run, name="Transcriber")
    transcriber_thread.daemon = True
    threads.append(transcriber_thread)

    tracker_thread = Thread(target=tracker.run, name="SoundTracker")
    tracker_thread.daemon = True
    threads.append(tracker_thread)

    executor_thread = Thread(target=executor.run, name="MovementExecutor")
    executor_thread.daemon = True
    threads.append(executor_thread)

    for t in threads:
        t.start()
        logger.info("Started thread: %s", t.name)

    # Start both audio subscriptions (after threads are running)
    audio_service.start()
    srp_audio.start()
    logger.info("Audio subscriptions active -- listening for wake words...")
    logger.info("Say 'Ginny' near the robot to trigger SRP-PHAT localization + movement")

    def shutdown(signum=None, frame=None):
        logger.info("Shutting down...")
        state.stop()
        try:
            audio_service.stop()
        except Exception:
            pass
        try:
            srp_audio.stop()
        except Exception:
            pass
        for t in threads:
            t.join(timeout=2)
        try:
            motion.rest()
            logger.info("Robot is asleep")
        except Exception:
            pass
        logger.info("All threads stopped. Goodbye.")
        # Force exit - qi framework spawns threads that ignore KeyboardInterrupt
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
