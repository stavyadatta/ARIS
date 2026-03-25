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
import os
import sys
sys.path.insert(0, "/workspace/diarization/DiariZen/pyannote-audio")
sys.path.insert(0, "/workspace/diarization/DiariZen")

# torch >=2.6 defaults weights_only=True which breaks pyannote/DiariZen/ultralytics
# checkpoint loading. Patch torch.load to default weights_only=False.
# Safe here since we trust all local model checkpoints.
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import grpc
import logging
from concurrent import futures

from speaker_service import SpeakerRecognitionManager
import grpc_communication.grpc_pb2_grpc as pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("speaker_server")

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


MODEL_DISPLAY = {
    "eres2netv2": ("ERes2NetV2 (voxceleb)", "voice_eres2netv2"),
    "titanet":    ("TitaNet Large (NeMo)",  "voice_titanet"),
    "redimnet":   ("ReDimNet B6 (IDRnD)",   "voice_redimnet"),
    "wavlm_ssl":  ("WavLM-MHFA (SSL-SV)",   "voice_wavlm_ssl"),
}


def serve(port=50051, max_workers=10, model="eres2netv2"):
    # Set env var so core_api/__init__.py picks up the model choice
    os.environ["SPEAKER_MODEL"] = model
    model_label, model_dir = MODEL_DISPLAY[model]

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    print(f"""
{CYAN}{BOLD}{'=' * 60}
   SPEAKER + FACE RECOGNITION SERVER
{'=' * 60}{RESET}

{BOLD}  Pipeline:{RESET}
    Face    InsightFace buffalo_l     {DIM}cuda:0{RESET}
    Voice   {model_label:<25}{DIM}cuda:1{RESET}
    Diar    DiariZen v2 (4-spk)       {DIM}cuda:2{RESET}

{BOLD}  Storage:{RESET}
    Faces   /workspace/database/face_db/
    Voices  /workspace/database/{model_dir}/

{DIM}  Loading models...{RESET}
""")

    pb2_grpc.add_SpeakerRecognitionServiceServicer_to_server(
        SpeakerRecognitionManager(),
        server
    )

    server.add_insecure_port(f"[::]:{port}")

    print(f"""
{GREEN}{BOLD}  Server ready on port {port}{RESET}
{DIM}  Waiting for client connections...
  Press Ctrl+C to stop.{RESET}
{CYAN}{'=' * 60}{RESET}
""")

    try:
        server.start()
        server.wait_for_termination()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}  Shutting down...{RESET}")
        server.stop(0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Speaker + Face Recognition Server")
    parser.add_argument("--model", choices=["eres2netv2", "titanet", "redimnet", "wavlm_ssl"],
                        default="eres2netv2",
                        help="Speaker verification model (default: eres2netv2)")
    parser.add_argument("--port", type=int, default=50051,
                        help="gRPC port (default: 50051)")
    args = parser.parse_args()
    serve(port=args.port, model=args.model)
