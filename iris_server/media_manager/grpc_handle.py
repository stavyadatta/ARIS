import os
import random
import cv2
import time
from queue import Empty as QueueEmpty
import wave
import traceback
import numpy as np
from google.protobuf.empty_pb2 import Empty

from core_api import FaceRecognition, WhisperSpeech2Text, ClipClassification
from executor import Executor
from reasoner import Reasoner
from utils import (
    ACTION_NONE,
    ACTION_SCRATCH_HEAD,
    G1_ACTION_MODE,
    g1_action_payload,
)
from grpc_pb2 import TextChunk, FaceBoundingBox, QueueRemoval
from grpc_pb2_grpc import MediaServiceServicer
from turn_timing import mark, record_first_moment, span, turn

# These are deliberately longer than a one-line error. They give G1's
# Scratch_head custom action time to read naturally while asking for a retry.
G1_LISTENING_FALLBACKS = (
    "I am sorry, I did not catch that clearly. Could you please say it once more for me?",
    "My listening was not very good just then. Would you mind repeating that a little more slowly?",
    "Excuse me, I missed part of what you said. Could you repeat it one more time, please?",
    "I am still learning to listen in a noisy room. Please say that again when you are ready.",
    "Sorry, I did not hear you clearly enough. Could you try that again for me, please?",
)

# Said aloud when something inside the server breaks -- an exception, not a
# misheard sentence. Deliberately different in kind from G1_LISTENING_FALLBACKS
# above: those ask for a repeat, because a repeat genuinely helps when the robot
# mishears. Repeating yourself does nothing about an exception, so these say so
# and point at a human instead. Telling someone to try again when trying again
# cannot work leaves them doing it forever.
#
# Rotating, for the same reason the refusals rotate: these are the replies most
# likely to repeat, and hearing one sentence twice makes the fault obvious.
G1_ERROR_REPLIES = (
    "I am sorry, something has gone wrong inside me. This one needs my "
    "engineers to take a look, so please let the team know.",
    "Apologies, my software has run into a problem I cannot fix myself. "
    "Would you mind telling someone from the team about it?",
    "Sorry, something is broken on my side and trying again will not help. "
    "Please ask the team to check my logs.",
    "Sorry, I am not working properly at the moment, and it is not "
    "because of anything you said. Someone technical will need to look at me.",
)

IMAGE_QUEUE_LEN = 50

# Chunks the executor marks 'default' are raw streamed speech; every other
# mode is already a structured payload and passes through untouched.
SPEECH_CHUNK_MODE = 'default'

VISION_STATE = "vision"

# Whisper returns a bare "." or "" on silence. A placeholder word keeps the
# reasoner on its normal path, where the prompt classifies it as bad input.
MIN_USABLE_TRANSCRIPTION_LEN = 2
SILENCE_PLACEHOLDER = "You"

# Served by image_viewer on :8003 for debugging what the pipeline just saw.
CURRENT_FRAME_PATH = "/workspace/display_imgs/some.jpg"

SAMPLE_WIDTHS_BY_ENCODING = {
    "PCM_8": 1,   # 8-bit audio
    "PCM_16": 2,  # 16-bit audio
    "PCM_24": 3,  # 24-bit audio
    "PCM_32": 4   # 32-bit audio
}

EMPTY_FACE_BBOX = FaceBoundingBox(x1=0, y1=0, x2=0, y2=0)

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

    def _transcribe(self, audio_img_item):
        transcription = WhisperSpeech2Text(audio_img_item)
        if len(transcription) < MIN_USABLE_TRANSCRIPTION_LEN:
            transcription = SILENCE_PLACEHOLDER
        print(f"Transcription: {transcription}")
        return transcription

    def _face_id_from_stream_votes(self, image):
        """Vote over recently streamed frames, then retry on this request's."""
        face_id = FaceRecognition.get_most_frequent_face_id()
        if face_id is not None:
            return face_id

        # The voting queues only hold StreamImages frames, which may be
        # absent or all rejected. The frame attached to this request is a
        # second chance, and is the one the person was actually in front of
        # while speaking.
        face_id = FaceRecognition.recognize_face_relaxed(image)
        if face_id is not None:
            print(f"[face_id] {face_id} via request frame")
        return face_id

    def _resolve_face_id(self, image, skip_face_validation):
        if skip_face_validation:
            face_id = FaceRecognition.recognize_face_relaxed(image)
        else:
            face_id = self._face_id_from_stream_votes(image)
        self._log_face_id(face_id, skip_face_validation)
        return face_id

    def _log_face_id(self, face_id, skip_face_validation):
        """Name the recognition path, since a miss silences the LLM entirely.

        A missing face_id makes the reasoner return "no face" before it ever
        reaches the LLM. Iris asking to be seen every turn means this fired.
        """
        if face_id is not None:
            print(f"[face_id] {face_id} relaxed={skip_face_validation}")
            return
        print(
            f"[face_id] none relaxed={skip_face_validation} "
            f"stream_frames_pending={FaceRecognition.face_img_queue.qsize()} "
            f"recognized_votes={list(FaceRecognition.face_id_queue)}"
        )

    def _reason_about(self, transcription, face_id, image):
        person_details = Reasoner(transcription, face_id)
        if person_details.get_attribute("state") == VISION_STATE:
            person_details.set_image(image)
        print(f"Resolved person state: {person_details.get_attribute('state')}")
        return person_details

    def _speech_chunk(self, reply):
        print(f"\n[g1_action] action={ACTION_NONE}")
        return (g1_action_payload(reply, ACTION_NONE), G1_ACTION_MODE)

    def _apology_chunk(self):
        """A spoken apology, in the ordinary reply contract.

        Errors used to travel as mode="error" carrying a stack trace, which the
        client could not voice -- the robot said the word "error" at whoever was
        standing there. A fault is not the person's fault and not something they
        can act on, so Iris says sorry in her own voice and the detail stays in
        the log.
        """
        reply = random.choice(G1_ERROR_REPLIES)
        print(f"[g1_action] apologising; action={ACTION_NONE}")
        return (g1_action_payload(reply, ACTION_NONE), G1_ACTION_MODE)

    def _listening_fallback_chunk(self, log_reason):
        """Ask for a repeat without inventing a physical action."""
        reply = random.choice(G1_LISTENING_FALLBACKS)
        print(f"\n[g1_action] {log_reason}; using fallback")
        print(f"[g1_action] action={ACTION_SCRATCH_HEAD}")
        return (g1_action_payload(reply, ACTION_SCRATCH_HEAD), G1_ACTION_MODE)

    def _spoken_reply_chunk(self, reply):
        if not reply.strip():
            # A stale/"silent" Neo4j state must not make the G1 appear
            # unresponsive during a spoken conversation.
            return self._listening_fallback_chunk("empty normal reply")
        return self._speech_chunk(reply)

    def _g1_conversation_chunks(self, api_response):
        """Fold executor output into the G1 reply/action contract.

        A normal spoken reply is deliberately an idle action rather than an
        implicit body command. Gesture APIs already emit their own g1_action
        payload, so those chunks pass through unchanged.
        """
        speech_parts = []
        emitted_structured_chunk = False

        for response_chunk in api_response:
            print(response_chunk.textchunk, end='', flush=True)
            if response_chunk.mode == SPEECH_CHUNK_MODE:
                speech_parts.append(response_chunk.textchunk)
                continue

            emitted_structured_chunk = True
            if speech_parts:
                yield self._speech_chunk(''.join(speech_parts))
                speech_parts = []
            yield (response_chunk.textchunk, response_chunk.mode)

        if speech_parts:
            yield self._spoken_reply_chunk(''.join(speech_parts))
        elif not emitted_structured_chunk:
            yield self._listening_fallback_chunk("executor yielded no reply")

    def _getting_response(self, audio_img_item, skip_face_validation=False):
        if audio_img_item is None:
            return None
        try:
            with span("transcribe"):
                transcription = self._transcribe(audio_img_item)

            image = audio_img_item.get("image_data")
            with span("write_debug_frame"):
                cv2.imwrite(CURRENT_FRAME_PATH, image)
            with span("resolve_face_id"):
                face_id = self._resolve_face_id(image, skip_face_validation)

            with span("reason"):
                person_details = self._reason_about(transcription, face_id, image)
            mark("route", person_details.get_attribute("state"))

            print("Executor response:")
            # Executor returns an unstarted generator, so this span is dispatch
            # only; the API's own work lands inside api_response below.
            with span("executor_dispatch"):
                api_response = Executor(person_details)
            with span("api_response"):
                yield from self._g1_conversation_chunks(api_response)

        except Exception as e:
            print(f"Error processing audio: {e}")
            traceback.print_exc()
            yield self._apology_chunk()

    def _save_request_audio(self, request):
        self.save_audio_to_file(
            audio_data=request.audio_data,
            sample_rate=request.sample_rate,
            num_channels=request.num_channels,
            sample_width=SAMPLE_WIDTHS_BY_ENCODING.get(request.audio_encoding),
            file_name=f"audio_{int(time.time())}.wav"
        )

    def _audio_img_item(self, request, image):
        return {
            "audio_data": request.audio_data,
            "sample_rate": request.sample_rate,
            "num_channels": request.num_channels,
            "encoding": request.audio_encoding,
            "description": request.audio_description,
            "image_data": image
        }

    def ProcessAudioImg(self, request, context):
        with turn("process_audio_img"):
            yield from self._process_audio_img(request)

    def _process_audio_img(self, request):
        chunks_out = 0
        try:
            if self.audio_save:
                self._save_request_audio(request)

            with span("decode_image"):
                image = self._decode_image_from_bytes(request.image_data)
            print("Image has been decoded I think")
            if image is None:
                print("Is the image coming as None")
                mark("outcome", "no_image")
                print("[g1_action] image decode failed; apologising aloud")
                payload, mode = self._apology_chunk()
                yield TextChunk(mode=mode, text=payload)
                return

            pipeline_response = self._getting_response(
                self._audio_img_item(request, image),
                skip_face_validation=request.skip_face_validation
            )
            for response_text, mode in pipeline_response:
                record_first_moment("first_chunk_at_ms")
                chunks_out += 1
                yield TextChunk(text=response_text, is_final=False, mode=mode)
            mark("chunks_out", chunks_out)

        except Exception as e:
            print("Error occurred while processing data: {}".format(
                traceback.format_exc()))
            mark("outcome", "error")
            # This is a generator: a returned value is discarded, so the
            # client would be handed a silent stream instead of the error.
            payload, mode = self._apology_chunk()
            yield TextChunk(mode=mode, text=payload)

    def StreamImages(self, request_iterator, context):
        """
            Handle the image streaming requests from the client
        """
        try:
            for request in request_iterator:
                if request.face_min_area > 0:
                    FaceRecognition.min_area = request.face_min_area

                image = self._decode_image_from_bytes(request.image_data)
                if image is not None:
                    # GetBbox reads the newest frame from here. Without this
                    # append the deque stays empty and every bbox request is
                    # answered with zeroes.
                    self.image_queue.append(image)
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
        if len(self.image_queue) < IMAGE_QUEUE_LEN:
            return EMPTY_FACE_BBOX

        bbox = FaceRecognition.get_face_box(self.image_queue[-1])
        if bbox is None:
            return EMPTY_FACE_BBOX

        x1, y1, x2, y2 = bbox
        return FaceBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    
    def _drain_pending_frames(self):
        """Empty the recognition queue without blocking on a racing producer."""
        face_img_queue = FaceRecognition.face_img_queue
        while True:
            try:
                face_img_queue.get_nowait()
            except QueueEmpty:
                return

    def ClearQueue(self, request, context):
        """Drop every frame still queued, so the next turn starts clean."""
        try:
            self._drain_pending_frames()
            self.image_queue.clear()
        except Exception as e:
            # The previous version reported success here: its early return
            # left the inner helper, not this method.
            print("Error while removing the queue ", e)
            return QueueRemoval(removed=False)

        print("The queues have been cleared")
        return QueueRemoval(removed=True)
