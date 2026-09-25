import grpc
from collections import deque
from concurrent import futures
from threading import Thread

from core_api import FaceRecognition, WhisperSpeech2Text
from media_manager import MediaManager, IMAGE_QUEUE_LEN
from secondary_channel import SecondaryGRPC
import grpc_communication.grpc_pb2_grpc as pb2_grpc

from image_viewer import image_serve


def warm_up_models():
    """Run one throwaway inference per model before accepting any request.

    Whisper and InsightFace both defer CUDA context creation and kernel
    autotuning to their first call. Measured, that put 1.7 s of InsightFace
    and 0.4 s of Whisper on whoever speaks first after a restart -- exactly
    the turn a demo audience sees.
    """
    try:
        WhisperSpeech2Text.warm_up()
        FaceRecognition.warm_up()
        print("Models warmed up")
    except Exception as e:
        print(f"Model warm-up failed, the first turn will be slower: {e}")


def serve():
    """
    Start the gRPC server and processing loop.
    """
    image_queue = deque(maxlen=IMAGE_QUEUE_LEN)
    img_serve_thread = Thread(
        target=image_serve,
        daemon=True
    )
    img_serve_thread.start()

    # Start the gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_MediaServiceServicer_to_server(
        MediaManager(image_queue), 
        server
    )
    pb2_grpc.add_SecondaryChannelServicer_to_server(
        SecondaryGRPC(),
        server
    )
    server.add_insecure_port("[::]:50051")
    warm_up_models()
    print("gRPC server running on port 50051...")
    try:
        server.start()
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("Shutting down server...")
        server.stop(0)

if __name__ == "__main__":
    serve()
