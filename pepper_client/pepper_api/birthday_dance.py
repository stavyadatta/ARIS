import time
import random


class BirthdayDance:
    """Plays the birthday song while Pepper performs randomised happy movements.

    Usage:
        bd = BirthdayDance()
        bd.perform(session)   # blocks until the routine finishes
    """

    HAPPY_TAGS = ["excited", "happy", "enthusiastic"]


    SONG_PATH = "/home/nao/birthday.wav"
    SONG_DURATION = 30  # seconds – adjust to match the actual file length

    def perform(self, session):
        """Play the birthday song and dance. Blocks until done."""
        audio_player = session.service("ALAudioPlayer")
        animation_player = session.service("ALAnimationPlayer")
        posture_service = session.service("ALRobotPosture")

        # Stand up straight before we begin
        posture_service.goToPosture("Stand", 0.5)

        # Start the song (non-blocking so we can dance over it)
        file_id = audio_player.loadFile(self.SONG_PATH)
        audio_player.play(file_id, _async=True)

        # Build a combined pool of all happy moves and shuffle for variety
        all_moves = list(self.HAPPY_TAGS) 

        end_time = time.time() + self.SONG_DURATION
        while time.time() < end_time:
            random.shuffle(all_moves)
            for move in all_moves:
                if time.time() >= end_time:
                    break
                try:
                    animation_player.runTag(move)
                except Exception:
                    pass
                time.sleep(0.5)

        # --- Song over: stop immediately and reset ---
        audio_player.stopAll()
        try:
            animation_player.stopAll()
        except Exception:
            pass
        posture_service.goToPosture("StandInit", 0.3)
