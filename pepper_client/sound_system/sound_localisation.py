import qi
import time
import argparse

class SoundLocalizer:
    def __init__(self, session, ip, port):
        self.session = session
        self.sound_localization = self.session.service("ALSoundLocalization")
        self.memory = self.session.service("ALMemory")
        self.motion = self.session.service("ALMotion")
        self.audio_recorder = self.session.service("ALAudioRecorder")
        self.is_running = False

    def start(self):
        self.sound_localization.subscribe("SoundLocated")
        self.is_running = True
        print("Sound localization started.")
        self.sound_localization.setParameter("Sensibility", 0.8)
        # self.start_recording()

    def stop(self):
        self.sound_localization.unsubscribe("SoundLocated")
        self.is_running = False
        self.stop_recording()
        print("Sound localization stopped.")

    def start_recording(self):
        """Starts recording the audio."""
        self.audio_recorder.stopMicrophonesRecording()
        self.audio_recorder.startMicrophonesRecording(
            "/workspace/Documents/ginnyMAN/pepper_client/sound_system/wav_files/sound.wav",
            "wav",
            16000,
            [0, 0, 1, 0]
        )
        print("Recording started.")

    def stop_recording(self):
        """Stops the audio recording."""
        self.audio_recorder.stopMicrophonesRecording()
        print("Recording stopped.")

    def get_sound_location(self):
        if not self.is_running:
            return None
        return self.memory.getData("ALSoundLocalization/SoundLocated")

    def move_head(self, azimuth):
        """Moves the robot's head towards the sound."""
        self.motion.setStiffnesses("Head", 1.0)
        self.motion.angleInterpolationWithSpeed("HeadYaw", azimuth, 0.1)
        time.sleep(2)
        self.motion.angleInterpolationWithSpeed("HeadYaw", 0.0, 0.1)
        self.motion.setStiffnesses("Head", 0.0)

    def move_body(self, azimuth):
        """Moves the robot's body towards the sound."""
        self.motion.setStiffnesses("Body", 1.0)
        self.motion.moveTo(0, 0, azimuth)
        time.sleep(2)
        self.motion.moveTo(0, 0, 0)
        self.motion.setStiffnesses("Body", 0.0)

    def move_head_and_body(self, azimuth):
        """Moves the robot's head and body towards the sound."""
        self.motion.setStiffnesses("Head", 1.0)
        self.motion.setStiffnesses("Body", 1.0)
        self.motion.angleInterpolationWithSpeed("HeadYaw", azimuth, 0.1)
        self.motion.moveTo(0, 0, azimuth)
        time.sleep(2)
        self.motion.angleInterpolationWithSpeed("HeadYaw", 0.0, 0.1)
        self.motion.moveTo(0, 0, 0)
        self.motion.setStiffnesses("Head", 0.0)
        self.motion.setStiffnesses("Body", 0.0)

    def run(self, with_head_movement=False, with_body_movement=False, with_head_and_body_movement=False):
        self.start()
        try:
            while True:
                sound_location = self.get_sound_location()
                if sound_location:
                    self.print_sound_location(sound_location)
                    azimuth = sound_location[1][0]
                    if with_head_movement:
                        self.move_head(azimuth)
                    if with_body_movement:
                        self.move_body(azimuth)
                    if with_head_and_body_movement:
                        self.move_head_and_body(azimuth)
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
            print("Sound localization finished.")

    def print_sound_location(self, data):
        if not data or len(data) < 2:
            return

        azimuth = data[1][0]
        elevation = data[1][1]
        confidence = data[1][2]
        energy = data[2]

        print("----------------------------------------")
        print("           SOUND DETECTED               ")
        print("----------------------------------------")
        print(f"Azimuth: {azimuth:.2f} rad")
        print(f"Elevation: {elevation:.2f} rad")
        print(f"Confidence: {confidence:.2f}")
        # print(f"Energy: {energy:.2f}")
        print("========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Please enter Pepper's IP address (and optional port number)")
    parser.add_argument("--ip", type=str, nargs='?', default="192.168.0.52")
    parser.add_argument("--port", type=int, nargs='?', default=9559)
    parser.add_argument("--head", action="store_true", help="Enable head movement")
    parser.add_argument("--body", action="store_true", help="Enable body movement")
    parser.add_argument("--headandbody", action="store_true", help="Enable head and body movement")
    args = parser.parse_args()

    pepper_connection_url = "tcp://" + args.ip + ":" + str(args.port)
    app = qi.Application(["SoundLocalizer", "--qi-url=" + pepper_connection_url])
    app.start()
    session = app.session

    sound_localizer = SoundLocalizer(session, args.ip, args.port)
    sound_localizer.run(with_head_movement=args.head, with_body_movement=args.body, with_head_and_body_movement=args.headandbody)
