import grpc
from collections import deque
from concurrent import futures
from threading import Thread

from media_manager import MediaManager, IMAGE_QUEUE_LEN
from secondary_channel import SecondaryGRPC
from speaker_service import SpeakerRecognitionManager
import grpc_communication.grpc_pb2_grpc as pb2_grpc

from image_viewer import image_serve


def serve():
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
    pb2_grpc.add_SpeakerRecognitionServiceServicer_to_server(
        SpeakerRecognitionManager(),
        server
    )

    server.add_insecure_port("[::]:50051")
    print("gRPC server running on port 50051...")
    try:
        server.start()
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("Shutting down server...")
        server.stop(0)


if __name__ == "__main__":
    serve()
