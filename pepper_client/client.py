"""
ARIS Pepper client.

Runs next to the robot (e.g. on a Raspberry Pi 5) and bridges Pepper with the
ARIS server over gRPC:

    * a background thread streams camera frames to the server for face
      re-identification,
    * the main loop records an utterance (front-mic energy gate), sends it with
      the latest frame to the server, and executes the streamed response:
      plain text is spoken with body-talk animations, JSON in mode
      "custom_movement" / "standard_movement" is executed as a movement.

Usage:
    python client.py --ip 192.168.0.52 --server 10.0.0.5:50051 --listen tcp://<this-host-ip>:9559
"""
import io
import os
import re
import sys
import json
import math
import time
import random
import logging
import argparse
import traceback
from collections import deque
from threading import Thread, Event

import cv2
import grpc
import numpy as np
import qi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "grpc_communication"))
from grpc_pb2 import AudioImgRequest, ImageStreamRequest  # noqa: E402
from grpc_pb2_grpc import MediaServiceStub  # noqa: E402

logging.basicConfig(filename="app.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Scale used when ARIS was evaluated; the custom-movement prompt on the server
# was tuned against it, so it is kept instead of the exact pi / 180.
DEG_TO_RAD = math.pi / 100

DANCES = [
    "asereje", "danceoftheknights", "danceofthereedflutes", "dango3kyodai",
    "Disco", "HeadBang", "hungarian", "jgangnamstyle", "la_bamba",
    "mika_relax", "walk_this_way",
]


MOVEMENT_MODES = ("custom_movement", "standard_movement")


class AudioManager(object):
    """
    qi service fed by ALAudioDevice. Records only once the front-mic energy
    crosses the threshold and stops after `max_below_thresh_loops` buffers
    below it.
    """

    def __init__(self, mic_threshold=370, max_below_thresh_loops=15):
        self.module_name = "AudioManager2"
        self.sample_rate = 16000
        self.channels = 1
        self.mic_threshold = mic_threshold
        self.max_below_thresh_loops = max_below_thresh_loops
        self.audio_data_buffer = io.BytesIO()

    def init_service(self, session):
        self.audio_service = session.service("ALAudioDevice")
        self.audio_service.enableEnergyComputation()

    def processRemote(self, nbOfChannels, nbOfSamplesByChannel, timeStamp, inputBuffer):
        current_energy = self.audio_service.getFrontMicEnergy()

        if current_energy > self.mic_threshold:
            self.below_threshold_count = 0
            self.first_high_thresh = True
            self.audio_data_buffer.write(inputBuffer)
        elif self.first_high_thresh:
            self.below_threshold_count += 1
            self.audio_data_buffer.write(inputBuffer)
            if self.below_threshold_count >= self.max_below_thresh_loops:
                self.isProcessingDone = True
                self.first_high_thresh = False
                self.below_threshold_count = 0

    def startProcessing(self):
        """Subscribe to the microphones and return one recorded utterance."""
        self.isProcessingDone = False
        self.audio_data_buffer = io.BytesIO()
        self.below_threshold_count = 0
        self.first_high_thresh = False

        self.audio_service.setClientPreferences(self.module_name, self.sample_rate, self.channels, 0)
        self.audio_service.subscribe(self.module_name)
        while not self.isProcessingDone:
            time.sleep(0.1)
        self.audio_service.unsubscribe(self.module_name)

        self.audio_data_buffer.seek(0)
        return self.audio_data_buffer.read(), self.sample_rate


class SpeechProcessor(object):
    """
    Turns the server's text stream into sentences (builder thread) and speaks
    them or executes movement JSON (speaker thread) while a third thread plays
    body-talk animations during speech.
    """

    def __init__(self, pepper):
        self.pepper = pepper
        self.sentence_queue = deque()
        self.current_sentence = ""
        self.is_running = True
        self.to_execute_movement_thread = True
        self.do_movement = Event()
        self.body_thread = Thread(target=self.body_movement)
        self.body_thread.start()

    def body_movement(self):
        movement_num = 1
        while self.to_execute_movement_thread:
            if self.do_movement.is_set():
                self.pepper.perform_body_speech(movement_num)
                movement_num = movement_num % 16 + 1
            else:
                time.sleep(0.1)

    def build_sentences(self, response_stream):
        try:
            for chunk in response_stream:
                sys.stdout.write(chunk.text)
                sys.stdout.flush()
                if chunk.mode == "error":
                    print("\nServer error: {}".format(chunk.text))
                    break

                if chunk.mode in MOVEMENT_MODES:
                    # Movement commands arrive as one complete JSON chunk
                    self.sentence_queue.append((chunk.text, chunk.mode))
                    continue

                self.current_sentence += chunk.text
                if re.search(r'[.!?]$', chunk.text.strip()):
                    self.sentence_queue.append((self.current_sentence.strip(), chunk.mode))
                    self.current_sentence = ""

                if not self.is_running:
                    break
            if self.current_sentence.strip():
                self.sentence_queue.append((self.current_sentence.strip(), "default"))
        except grpc.RpcError as e:
            print("gRPC LLM response error: {} - {}".format(e.code(), e.details()))

    def execute_response(self):
        while self.is_running or self.sentence_queue:
            if not self.sentence_queue:
                time.sleep(0.05)
                continue
            sentence, mode = self.sentence_queue.popleft()
            if mode == "custom_movement":
                self.pepper.custom_movement(sentence)
            elif mode == "standard_movement":
                self.pepper.standard_movement(sentence)
            elif sentence:
                self.do_movement.set()
                self.pepper.say(sentence)
        self.do_movement.clear()


class Pepper(object):
    def __init__(self, pepper_connection_url, listen_url, stub, mic_threshold, silence_loops, volume):
        self.stub = stub

        app = qi.Application(["AudioManager2", "--qi-url=" + pepper_connection_url])
        app.start()
        self.session = app.session
        print("Connected to the Pepper!")

        # ALAudioDevice calls processRemote() on this process, so the robot
        # must be able to reach this endpoint.
        self.session.listen(listen_url)
        self.audio_manager = AudioManager(mic_threshold, silence_loops)
        self.session.registerService("AudioManager2", self.audio_manager)
        self.audio_manager.init_service(self.session)
        self.audio_manager.audio_service.setOutputVolume(volume)

        self.life_service = self.session.service("ALAutonomousLife")
        self.motion_service = self.session.service("ALMotion")
        self.posture_service = self.session.service("ALRobotPosture")
        self.speech_service = self.session.service("ALTextToSpeech")
        self.animation_service = self.session.service("ALAnimationPlayer")
        self.leds_service = self.session.service("ALLeds")
        self.camera_service = self.session.service("ALVideoDevice")

        self.speech_service.setLanguage("English")
        self.life_service.setAutonomousAbilityEnabled("All", False)

        # Top camera, resolution 5, RGB (colorspace 11), 30 fps.
        self.video_client = self.camera_service.subscribeCamera("cam", 0, 5, 11, 30)
        self.camera_service.setAllParametersToDefault(0)
        self.camera_service.setParameter(0, 8, 1)  # vertical flip

        self.motion_service.setStiffnesses("Head", 1.0)
        self.motion_service.wakeUp()
        self.posture_service.goToPosture("StandInit", 0.5)

        self.default_posture()

    # ---------- robot primitives ----------
    def set_eyes(self, color):
        self.leds_service.fadeRGB("FaceLeds", color, 0.01)

    def say(self, text):
        self.speech_service.say(text)

    def perform_body_speech(self, animation_num):
        self.animation_service.run(
            "animations/Stand/BodyTalk/Speaking/BodyTalk_{}".format(animation_num)
        )

    def default_posture(self):
        """Resets the head to the default resting position."""
        self.motion_service.setAngles(["HeadYaw", "HeadPitch"], [0.0, -0.40], 0.2)

    def custom_movement(self, llm_response):
        action_items = json.loads(llm_response).get("action_list", [])
        for action in action_items:
            try:
                self.motion_service.angleInterpolationWithSpeed(
                    action.get("joint_name"),
                    action.get("angle") * DEG_TO_RAD,
                    action.get("speed"),
                )
            except Exception as e:
                print("An error occurred while moving the arm {}".format(e))
        self.posture_service.goToPosture("StandInit", 0.2)
        time.sleep(2)

    def standard_movement(self, llm_response):
        # Dances come from a Choregraphe package ("hapy_dance") installed on the
        # robot; fall back to Pepper's built-in "happy" animations without it.
        dance_type = random.choice(DANCES)
        print("Now performing ", dance_type)
        try:
            self.animation_service.run("hapy_dance/{}".format(dance_type))
        except Exception:
            self.animation_service.runTag("happy")

    # ---------- perception ----------
    def get_audio(self):
        self.set_eyes(0x000000FF)  # blue: listening
        audio_data, samplerate = self.audio_manager.startProcessing()
        self.set_eyes(0x00FF0000)  # red: processing
        return audio_data, samplerate

    def make_img_compatible(self):
        raw_image = self.camera_service.getImageRemote(self.video_client)
        self.camera_service.releaseImage(self.video_client)
        width, height = raw_image[0], raw_image[1]
        rgb = np.frombuffer(bytes(raw_image[6]), dtype=np.uint8).reshape(height, width, 3)
        # RGB -> BGR for OpenCV, then flip to match the server's expectations
        return cv2.flip(rgb[:, :, ::-1], 0)

    def img_stream(self):
        cv2_image = self.make_img_compatible()
        height, width, _ = cv2_image.shape
        _, image_data = cv2.imencode(".jpg", cv2_image)

        request = ImageStreamRequest(
            image_data=image_data.tobytes(),
            image_format="JPEG",
            image_width=width,
            image_height=height,
            image_description="Captured pepper image",
        )
        try:
            self.stub.StreamImages(iter([request]))
        except grpc.RpcError as e:
            logger.error("Failed to stream image: {}".format(e.details()))
        time.sleep(0.1)

    def capture_and_stream_images(self):
        while True:
            try:
                self.img_stream()
            except TypeError:
                # getImageRemote occasionally returns None
                time.sleep(0.1)

    # ---------- interaction ----------
    def process_server_response(self, server_response_stream):
        speech_processor = SpeechProcessor(self)
        builder_thread = Thread(target=speech_processor.build_sentences, args=(server_response_stream,))
        speaker_thread = Thread(target=speech_processor.execute_response)

        builder_thread.start()
        speaker_thread.start()

        builder_thread.join()
        speech_processor.is_running = False
        speaker_thread.join()
        speech_processor.to_execute_movement_thread = False
        speech_processor.body_thread.join()
        self.default_posture()

    def main(self):
        audio_data, sample_rate = self.get_audio()
        try:
            last_frame = self.make_img_compatible()
        except TypeError:
            print("No camera frame available, skipping turn")
            return

        height, width, _ = last_frame.shape
        _, image_data = cv2.imencode(".jpg", last_frame)
        request = AudioImgRequest(
            audio_data=audio_data,
            sample_rate=sample_rate,
            num_channels=1,
            audio_encoding="PCM_16",
            audio_description="Audio captured by Pepper",
            image_data=image_data.tobytes(),
            image_format="JPEG",
            image_width=width,
            image_height=height,
            api_task="Captured Pepper",
        )
        try:
            server_response_stream = self.stub.ProcessAudioImg(request)
            self.process_server_response(server_response_stream)
        except grpc.RpcError as e:
            print("gRPC in sending audio error: {} - {}".format(e.code(), e.details()))

    def close(self):
        print("Shutting down Pepper services...")
        try:
            self.camera_service.unsubscribe(self.video_client)
            self.leds_service.reset("FaceLeds")
        except Exception:
            pass
        self.session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARIS Pepper client")
    parser.add_argument("--ip", default=os.getenv("PEPPER_IP", "192.168.0.52"), help="Pepper's IP address")
    parser.add_argument("--port", type=int, default=int(os.getenv("PEPPER_PORT", "9559")))
    parser.add_argument("--server", default=os.getenv("ARIS_SERVER", "localhost:50051"),
                        help="host:port of the ARIS gRPC server")
    parser.add_argument("--listen", default=os.getenv("CLIENT_LISTEN_URL", "tcp://0.0.0.0:9559"),
                        help="qi endpoint on this machine that Pepper streams microphone audio to")
    parser.add_argument("--mic-threshold", type=float, default=370,
                        help="front-mic energy that counts as speech")
    parser.add_argument("--silence-loops", type=int, default=15,
                        help="audio buffers below the threshold that end an utterance")
    parser.add_argument("--volume", type=int, default=int(os.getenv("PEPPER_VOLUME", "40")),
                        help="speaker volume, 0-100")
    args = parser.parse_args()

    channel = grpc.insecure_channel(args.server)
    stub = MediaServiceStub(channel)
    p = Pepper("tcp://{}:{}".format(args.ip, args.port), args.listen, stub,
               args.mic_threshold, args.silence_loops, args.volume)

    image_thread = Thread(target=p.capture_and_stream_images)
    image_thread.daemon = True
    image_thread.start()

    try:
        while True:
            p.main()
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    except Exception as e:
        print("An error occurred:", e)
        traceback.print_exc()
    finally:
        p.close()
