import grpc
from collections import deque
from concurrent import futures
from threading import Thread

from media_manager import MediaManager, IMAGE_QUEUE_LEN
import grpc_communication.grpc_pb2_grpc as pb2_grpc

from image_viewer import image_serve

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
