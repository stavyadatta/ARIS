import grpc
import wave
import cv2
import sys
import io
import logging
import numpy as np
import pyaudio
import pyttsx3  # <-- Text-to-speech library
from queue import Queue  # <-- For storing text chunks
from threading import Thread
from google.protobuf.empty_pb2 import Empty

from grpc_communication.grpc_pb2 import AudioImgRequest, ImageStreamRequest
from grpc_communication.grpc_pb2_grpc import MediaServiceStub

# -----------------------------------------------------------
# Logger Configuration
# -----------------------------------------------------------
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# -----------------------------------------------------------
# Global Variables
# -----------------------------------------------------------
THRESHOLD = 1000000
SILENCE_FRAMES = 10
CHUNK = 1024
RATE = 16000
CHANNELS = 1

# This will store the last captured webcam frame
static_img = None

# A queue to store the streamed text chunks
text_queue = Queue()

# -----------------------------------------------------------
# Audio Recording Functions
# -----------------------------------------------------------
def wait_for_speech_start(stream):
    """
    Wait until the user starts speaking (energy > THRESHOLD).
    Returns the first chunk of speech data when speech is detected.
    """
    logger.info("Waiting for speech...")
    while True:
        data = stream.read(CHUNK)
        audio_samples = np.frombuffer(data, dtype=np.int16)
        energy = np.sum(audio_samples.astype(np.int32) ** 2) / len(audio_samples)
        if energy > THRESHOLD:
            logger.info("Speech detected, starting recording...")
            return [data]  # Return the first chunk of speech data

def record_until_silence():
    """
    Record audio from the microphone starting when user speaks above a threshold.
    Stop after detecting SILENCE_FRAMES consecutive silent frames.
    Returns the recorded PCM (raw) data.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    # Wait until user starts speaking
    frames = wait_for_speech_start(stream)

    silent_count = 0
    while True:
        data = stream.read(CHUNK)
        frames.append(data)

        audio_samples = np.frombuffer(data, dtype=np.int16)
        energy = np.sum(audio_samples.astype(np.int32) ** 2) / len(audio_samples)

        if energy < THRESHOLD:
            silent_count += 1
        else:
            silent_count = 0

        if silent_count >= SILENCE_FRAMES:
            logger.info("Silence detected. Stopping recording.")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    return b''.join(frames)

def convert_pcm_to_wav_bytes(pcm_data, sample_rate, num_channels, sample_width=2):
    """
    Convert raw PCM data to a WAV-formatted byte string.
    """
    wav_buffer = io.BytesIO()
    wf = wave.open(wav_buffer, 'wb')
    wf.setnchannels(num_channels)
    wf.setsampwidth(sample_width)  # 2 bytes (16-bit)
    wf.setframerate(sample_rate)
    wf.writeframes(pcm_data)
    wf.close()
    return wav_buffer.getvalue()

# -----------------------------------------------------------
# Webcam Image Capture
# -----------------------------------------------------------
def capture_webcam_image(cap):
    """
    Capture a single frame from the webcam.
    Using a global variable static_img to store the last frame.
    """
    global static_img
    return static_img

# -----------------------------------------------------------
# gRPC Sending Functions
# -----------------------------------------------------------
def send_audio_image_to_grpc(audio_wav_data, sample_rate, num_channels, last_frame, stub):
    """
    Send the audio and the last captured webcam frame to the gRPC server.
    """
    try:
        # Encode the last frame as JPEG
        _, image_data = cv2.imencode('.jpg', last_frame)
        height, width, _ = last_frame.shape

        # Determine audio encoding
        audio_encoding = "PCM_16"

        # Create AudioImgRequest message
        logger.debug("Creating AudioImgRequest...")
        request = AudioImgRequest(
            audio_data=audio_wav_data,
            sample_rate=sample_rate,
            num_channels=num_channels,
            audio_encoding=audio_encoding,
            audio_description="Recorded audio data",
            image_data=image_data.tobytes(),
            image_format="JPEG",
            image_width=width,
            image_height=height,
            api_task="Captured webcam image"
        )

        # Call the gRPC method
        logger.info("Sending data to gRPC server...")
        response = stub.SendAudioImg(request)
        logger.info("SendAudioImg response: {} - {}".format(response.status, response.message))
    except Exception as e:
        logger.error("Error: Failed to send media via gRPC. Error: %s", e)

# -----------------------------------------------------------
# New Feature #1: Store streamed words in a queue
# -----------------------------------------------------------
def store_words_in_queue(text, queue_object):
    """
    Store the text chunk into a queue for further processing (e.g., TTS).
    """
    queue_object.put(text)

# -----------------------------------------------------------
# New Feature #2: Text-to-Speech from a queue
# -----------------------------------------------------------
# def speak_text_from_queue(queue_object):
#     """
#     Uses pyttsx3 to speak out all the text that has been accumulated in the queue.
#     """
#     engine = pyttsx3.init()  # Initialize the TTS engine
#
#     # Optionally, configure voice rate, volume, etc.
#     # engine.setProperty('rate', 150)
#     # engine.setProperty('volume', 1.0)
#
#     # Collect all text segments from the queue
#     all_text_segments = []
#     while not queue_object.empty():
#         segment = queue_object.get()
#         all_text_segments.append(segment)
#
#     # Combine all segments into one speech string
#     full_text = "".join(all_text_segments)
#     if full_text.strip():
#         logger.info("Speaking text: '%s'", full_text)
#         engine.say(full_text)
#         engine.runAndWait()

def speak_text_from_queue(queue_object):
    """
    Uses pyttsx3 to speak out all the text that has been accumulated in the queue.
    This call will BLOCK until the text is fully spoken.
    """
    engine = pyttsx3.init()  # Initialize the TTS engine
    
    # Optionally configure engine properties
    # engine.setProperty('rate', 150)
    # engine.setProperty('volume', 1.0)

    # Collect all text segments from the queue
    all_text_segments = []
    while not queue_object.empty():
        segment = queue_object.get()
        all_text_segments.append(segment)

    # Combine all segments into one speech string
    full_text = "".join(all_text_segments)
    if full_text.strip():
        logger.info("Speaking text (BLOCKING): '%s'", full_text)
        engine.say(full_text)
        engine.runAndWait()  # Blocks until speaking is done
        logger.info("Done speaking text.")

        # If you find that TTS might still be cutting off,
        # you can forcibly wait a tiny bit more (usually not needed):
        # import time
        # time.sleep(1)

# -----------------------------------------------------------
# gRPC Receiving Function (Where the words are streamed)
# -----------------------------------------------------------
def receive_llm_response(stub, queue_object):
    """
    Receive streamed LLM response from the gRPC server.
    Write partial chunks to stdout and store them in a queue.
    """
    request = Empty()
    try:
        response_stream = stub.LLmResponse(request)
        logger.info("Receiving streamed text chunks:")
        for chunk in response_stream:
            # Write partial chunk to stdout
            sys.stdout.write(chunk.text)
            sys.stdout.flush()

            # Store chunk in queue
            store_words_in_queue(chunk.text, queue_object)

            if chunk.is_final:
                logger.info("Final chunk received")
                sys.stdout.write("\n[Final chunk received]\n")
                sys.stdout.flush()
                break
    except grpc.RpcError as e:
        logger.error("gRPC error: %s - %s", e.code(), e.details())

# -----------------------------------------------------------
# Continuously Capture and Stream Images to gRPC Server
# -----------------------------------------------------------
def capture_and_stream_images(stub, cap):
    """
    Continuously capture frames from the webcam and stream them to the gRPC server.
    """
    if not cap.isOpened():
        logger.error("Could not open webcam")
        raise Exception("Could not open webcam")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture image")
                continue

            global static_img
            static_img = frame  # Update global frame

            # Encode as JPEG
            _, image_data = cv2.imencode('.jpg', frame)
            height, width, _ = frame.shape

            request = ImageStreamRequest(
                image_data=image_data.tobytes(),
                image_format="JPEG",
                image_width=width,
                image_height=height,
                image_description="Captured webcam image"
            )

            # Send the image stream request
            try:
                stub.StreamImages(iter([request]))
            except grpc.RpcError as e:
                logger.error(f"Failed to stream image: {e.details()}")

    except KeyboardInterrupt:
        logger.info("Stopping image streaming...")
    finally:
        cap.release()

# -----------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------
def main():
    # Set up gRPC channel and stub
    channel = grpc.insecure_channel("172.27.72.27:50051")
    stub = MediaServiceStub(channel)
    cap = cv2.VideoCapture(2)  # Open the default camera; adjust index if needed

    # Start a separate thread to continuously capture and stream images
    image_thread = Thread(target=capture_and_stream_images, args=(stub, cap), daemon=True)
    image_thread.start()

    try:
        while True:
            # 1. Record audio until silence
            pcm_data = record_until_silence()

            # 2. Convert PCM data to WAV (in-memory)
            audio_wav_data = convert_pcm_to_wav_bytes(pcm_data, RATE, CHANNELS)

            # 3. Capture the latest frame from webcam
            last_frame = capture_webcam_image(cap)

            # 4. Send audio+image data to the gRPC pipeline
            send_audio_image_to_grpc(audio_wav_data, RATE, CHANNELS, last_frame, stub)

            # 5. Receive LLM response (store partial text in queue)
            receive_llm_response(stub, text_queue)

            # 6. Speak out all text collected in the queue
            # speak_text_from_queue(text_queue)

            logger.info("Finished one cycle. Waiting for next speech...")

    except KeyboardInterrupt:
        logger.info("\nShutting down... Goodbye!")
        image_thread.join()


if __name__ == "__main__":
    main()

