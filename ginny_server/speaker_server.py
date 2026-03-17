"""
Standalone server for Face + Speaker Recognition.

Usage:
    python speaker_server.py

Starts ONLY:
    - FaceRecognition (InsightFace buffalo_l on cuda:0)
    - SpeakerRecognition (ERes2NetV2 on cuda:1)
    - DiariZen diarization (on cuda:1)
    - SpeakerRecognitionService gRPC on port 50051

Does NOT start: Whisper, LLMs, Reasoner, Executor, MediaService, SecondaryChannel.

Client sends audio + camera frames → server identifies face + voice → returns results.
"""
# MUST be first — set paths for vendored pyannote-audio + DiariZen
# before any other imports trigger the import chain
import sys
sys.path.insert(0, "/workspace/diarization/DiariZen/pyannote-audio")
sys.path.insert(0, "/workspace/diarization/DiariZen")

import grpc
import logging
from concurrent import futures

from speaker_service import SpeakerRecognitionManager
import grpc_communication.grpc_pb2_grpc as pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("speaker_server")


def serve(port=50051, max_workers=10):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    logger.info("Initializing Face + Speaker Recognition pipeline...")
    logger.info("  Face:  InsightFace buffalo_l (cuda:0)")
    logger.info("  Voice: ERes2NetV2 192-dim (cuda:1)")
    logger.info("  Diar:  DiariZen wavlm-large (cuda:1)")
    logger.info("  Face DB:  /workspace/database/face_db/")
    logger.info("  Voice DB: /workspace/database/voice_db/")

    pb2_grpc.add_SpeakerRecognitionServiceServicer_to_server(
        SpeakerRecognitionManager(),
        server
    )

    server.add_insecure_port(f"[::]:{port}")
    logger.info(f"Server running on port {port}")
    logger.info("Service: SpeakerRecognitionService (face + voice)")

    try:
        server.start()
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop(0)


if __name__ == "__main__":
    serve()
