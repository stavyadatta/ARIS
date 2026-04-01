import cv2
import grpc
import time
import logging
import traceback 
import numpy as np
from PIL import Image
from io import BytesIO
from threading import Thread
from google.protobuf.empty_pb2 import Empty

from server_api import PepperClientAPI as Pepper
from grpc_communication.grpc_pb2 import AudioImgRequest, ImageStreamRequest
from grpc_communication.grpc_pb2_grpc import MediaServiceStub
from utils import SpeechProcessor, is_zero_list, get_vh_axis

from movement import CustomMovementManager
from button_frontend import FaceArea_UI

logging.basicConfig(filename="app.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PepperManager():
    def __init__(self, grpc_stub):
        self.pepper = Pepper()
        self.stub = grpc_stub

    def capture_and_stream_images(self):
        try:
            while True:
                cv2_image = self.pepper.make_img_compatible()
                height, width, _ = cv2_image.shape
                _, image_data = cv2.imencode(".jpg", cv2_image)

                request = ImageStreamRequest(
                    image_data=image_data.tobytes(),
                    image_format="JPEG",
                    image_width=width,
                    image_height=height,
                    image_description="Captured pepper image",
                    face_min_area=FaceArea_UI.peek_face_min_area()
                )

                # Send the image stream request
                try:
                    self.stub.StreamImages(iter([request]))
                except grpc.RpcError as e:
                    logger.error("Failed to stream image: {}".format(e.details()))
        except KeyboardInterrupt:
            logger.info("Stopping image streaming...")
            raise KeyboardInterrupt()
        except TypeError:
            print("Running the capture stream again")
            self.capture_and_stream_images()

    def head_management(self):
        try:
            cv2_image = self.pepper.make_img_compatible()
            img_shape = cv2_image.shape
            request = Empty()
            person_missing = 0
            while True:
                time.sleep(1)
                bbox = self.stub.GetBbox(request)
                box = [bbox.x1, bbox.y1, bbox.x2, bbox.y2]
                if not is_zero_list(box):
                    person_missing += 1
                    vertical_ratio, horizontal_ratio = get_vh_axis(box, img_shape)

                    self.pepper.rotate_head(vertical_ratio=float(vertical_ratio), 
                                            horizontal_ratio=float(horizontal_ratio))
                elif is_zero_list(box) and person_missing > 30:
                    person_missing = 0
                    self.pepper.rotate_head_abs()
        except KeyboardInterrupt:
            logger.info("Stopping the head management")
            raise KeyboardInterrupt

        except Exception as e:
            print("Some error in head_management, donot know ", e)
            traceback.print_exc()


    def send_audio(self):
        audio_data, sample_rate = self.pepper.get_audio()
        try:
            last_frame = self.pepper.make_img_compatible()
        except TypeError:
            print("Send audio was not receiving last frame")
            self.send_audio()
        try:
            height, width, _ = last_frame.shape
            _, image_data = cv2.imencode(".jpg", last_frame)
            request = AudioImgRequest(
                audio_data=audio_data,
                sample_rate=sample_rate,
                num_channels=1,  # Assuming mono audio
                audio_encoding="PCM_16",
                audio_description="Audio captured by Pepper",
                image_data=image_data.tobytes(),
                image_format="JPEG",
                image_width=width,
                image_height=height,
                image_description="Captured Pepper"
            )

            # Send the request to the gRPC server
            try:
                response = self.stub.SendAudioImg(request)
            except grpc.RpcError as e:
                print("gRPC in sending audio error: {} - " \
                "{}".format(e.code(), e.details()))

            time.sleep(1)  # Adjust delay as needed
        except UnboundLocalError as e:
            print("Unbounded local error occuring")
            traceback.print_exc()
            self.send_audio()

    def receive_llm_response(self):
        speech_processor = SpeechProcessor(self.pepper.say_text)
        
        request = Empty()
        response_stream = self.stub.LLmResponse(request)


        builder_thread = Thread(
            target=speech_processor.build_sentences, 
            args=(response_stream,)
        )
        speaker_thread = Thread(target=speech_processor.execute_response)

        builder_thread.start()
        speaker_thread.start()

        # Wait for threads to complete
        builder_thread.join()
        speech_processor.is_running = False
        speaker_thread.join()

        if speech_processor.movement:
            for chunk in response_stream:
                CustomMovementManager(chunk, self.pepper)


if __name__ == "__main__":
    channel = grpc.insecure_channel("172.27.72.27:50051")
    stub = MediaServiceStub(channel)
    p = PepperManager(stub)

    image_thread = Thread(target=p.capture_and_stream_images, args=()) 
    image_thread.daemon = True
    image_thread.start()


    head_thread = Thread(target=p.head_management, args=())
    head_thread.daemon = True
    head_thread.start()
    
    try:
        while True:
            p.send_audio()
            p.receive_llm_response()
    except KeyboardInterrupt:
        print("Program interrupted by user.")
        image_thread.join()
        head_thread.join()
        exit()
    except Exception as e:
        # Ensure resources are cleaned up
        print("the p close is getting called because of the following issues ", e)
        traceback.print_exc()
