import os
import cv2
import queue
import json
import time
import wave
import traceback
import numpy as np
from google.protobuf.empty_pb2 import Empty

from core_api import FaceRecognition, WhisperSpeech2Text, ClipClassification
from executor import Executor
from reasoner import Reasoner
from grpc_pb2 import AudioImgResponse, TextChunk, FaceBoundingBox, QueueRemoval
from grpc_pb2_grpc import MediaServiceServicer

IMAGE_QUEUE_LEN = 50

class MediaManager(MediaServiceServicer):
    def __init__(self, image_queue, audio_save=False):
        super().__init__()
        self.audio_save = audio_save
        self.image_queue = image_queue

        if self.audio_save:
            self.save_directory = "./recordings_stavya/"
            os.makedirs(self.save_directory, exist_ok=True)

    def save_audio_to_file(self, audio_data, sample_rate, num_channels, sample_width, file_name):
        file_path = os.path.join(self.save_directory, file_name)
        try:
            with wave.open(file_path, 'wb') as wave_file:
                wave_file.setnchannels(num_channels)
                wave_file.setsampwidth(sample_width)
                wave_file.setframerate(sample_rate)
                wave_file.writeframes(audio_data)
            print(f"Audio saved to {file_path} with header")
        except Exception as e:
            print(f"Error saving audio to file: {e}")
            raise

    def _decode_image_from_bytes(self, image_bytes):
        """
        Decode image bytes received in AudioImgRequest into a NumPy array usable by OpenCV.

        Args:
            image_bytes (bytes): The raw image data in bytes.

        Returns:
            np.ndarray: Decoded image as a NumPy array.
        """
        try:
            # Convert bytes to a 1D NumPy array
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            # Decode the image array into a format usable by OpenCV
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode the image from bytes.")
            return image
        except Exception as e:
            print("Error decoding image: {}".format(e))
            return None

    def _getting_response(self, audio_img_item, skip_face_validation=False):
        if audio_img_item is None:
            return None
        try:
            transcription = WhisperSpeech2Text(audio_img_item)
            if len(transcription) < 2:
                transcription= "You"
            print(f"Transcription: {transcription}")
            if len(transcription) < 2:
                # If the transcription is less than 2 characters
                # it will send a word which will automatically be
                # categorised as bad input
                transcription = "You"

            # Get the face information
            image = audio_img_item.get("image_data")
            cv2.imwrite("/workspace/display_imgs/some.jpg", image)
            if skip_face_validation:
                face_id = FaceRecognition.recognize_face_relaxed(image)
            else:
                face_id = FaceRecognition.get_most_frequent_face_id()

            person_details = Reasoner(transcription, face_id)
            if person_details.get_attribute("state") == "vision":
                person_details.set_image(image)

            print("Executor response:")
            response = Executor(person_details)
            mode = 'default'
            for response_chunk in response:
                mode = response_chunk.mode
                response_text = response_chunk.textchunk
                print(response_text, end='', flush=True)
                yield (response_text, mode)

        except Exception as e:
            print(f"Error processing audio: {e}")
            traceback.print_exc()
            yield ("error", "error")

    def ProcessAudioImg(self, request, context):
        try:
            file_name = f"audio_{int(time.time())}.wav"  # Unique file name using timestamp
            encoding_map = {
                "PCM_8": 1,   # 8-bit audio
                "PCM_16": 2,  # 16-bit audio
                "PCM_24": 3,  # 24-bit audio
                "PCM_32": 4   # 32-bit audio
            }
            sample_width = encoding_map.get(request.audio_encoding)
            if self.audio_save:
                self.save_audio_to_file(
                    audio_data=request.audio_data,
                    sample_rate=request.sample_rate,
                    num_channels=request.num_channels,
                    sample_width=sample_width,
                    file_name=file_name
                )

            image_bytes = request.image_data
            image = self._decode_image_from_bytes(image_bytes)
            print("Image has been decoded I think")
            if image is None:
                print("Is the image coming as None")
                yield TextChunk(
                    mode="error",
                    text="The image came out as None"
                )
                return
            audio_img_item = {
                "audio_data": request.audio_data,
                "sample_rate": request.sample_rate,
                "num_channels": request.num_channels,
                "encoding": request.audio_encoding,
                "description": request.audio_description,
                "image_data": image
            }
            pipeline_response = self._getting_response(
                audio_img_item,
                skip_face_validation=request.skip_face_validation
            )
            for resp in pipeline_response:
                response_text = resp[0]
                mode = resp[1]
                yield TextChunk(text=response_text, is_final=False, mode=mode)

        except Exception as e:
            error_trace = traceback.format_exc()
            print("Error occurred while processing data: {}".format(error_trace))

            return TextChunk(
                mode="error",
                text=f"Some error occured {e}"
            )

    def StreamImages(self, request_iterator, context):
        """
            Handle the image streaming requests from the client
        """
        try:
            for request in request_iterator:
                image = self._decode_image_from_bytes(request.image_data)
                if image is not None:
                    # Add image to the Face and Clip queues
                    # self.image_queue.append(image)
                    FaceRecognition.add2face_img_queue(image)
                    # ClipClassification.add2clip_img_queue(image)

        except Exception as e:
            traceback.print_exc()
        return Empty()

    def GetBbox(self, request, context):
        """
            From the image queue runs a face detector, gets the first bbox and then 
            returns the FaceBoundingBox
        """
        if not self.image_queue:
            return FaceBoundingBox(
                x1=0,
                y1=0,
                x2=0,
                y2=0
            )

        if len(self.image_queue) < IMAGE_QUEUE_LEN:
            return FaceBoundingBox(
                x1=0,
                y1=0,
                x2=0,
                y2=0
            )
        earliest_image = self.image_queue[-1]

        bbox = FaceRecognition.get_face_box(earliest_image)
        if bbox is None:
            return FaceBoundingBox(
                x1=0,
                y1=0,
                x2=0,
                y2=0
            )

        x1, y1, x2, y2 = bbox
        face_bbox = FaceBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

        return face_bbox
    
    def ClearQueue(self, request, context):
        def empty_queue(q):
            try:
                while not q.empty():
                    q.get()
                    q.task_done()
            except Exception as e:
                print("Error while removing the queue ", e)
                return QueueRemoval(
                    removed=False
                )

        empty_queue(self.audio_img_queue)
        empty_queue(self.llama_response_queue)
        print("The ques have been cleaned for both")

        return QueueRemoval(
            removed=True
        )
