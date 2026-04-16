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

import re
import grpc
import logging
import datetime
from concurrent import futures

import grpc_communication.grpc_pb2_grpc as pb2_grpc


# ----------------------------------------------------------------------------
# Logging: stream to stdout (with ANSI colors) AND tee to a log file
# (with ANSI codes stripped). Log files live in ginny_server/logs/.
# ----------------------------------------------------------------------------
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(
    _LOG_DIR,
    f"speaker_server_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class _PlainFormatter(logging.Formatter):
    """Formatter that strips ANSI escape sequences for the file handler."""
    def format(self, record):
        msg = super().format(record)
        return _ANSI_RE.sub("", msg)


_root = logging.getLogger()
_root.setLevel(logging.INFO)
# Wipe any handlers a transitive import may have installed (e.g. via
# logging.basicConfig elsewhere) so we control the output cleanly.
for _h in list(_root.handlers):
    _root.removeHandler(_h)

_stream_handler = logging.StreamHandler()
_stream_handler.setLevel(logging.INFO)
_stream_handler.setFormatter(logging.Formatter("%(message)s"))
_root.addHandler(_stream_handler)

_file_handler = logging.FileHandler(_LOG_FILE, mode="w", encoding="utf-8")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(_PlainFormatter("%(asctime)s  %(name)s  %(message)s",
                                            datefmt="%H:%M:%S"))
_root.addHandler(_file_handler)

logger = logging.getLogger("speaker_server")
logger.info(f"Log file: {_LOG_FILE}")

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


def serve(port=50051, max_workers=10, model="eres2netv2", braid=False, no_asd=False):
    # Set env var so core_api/__init__.py picks up the model choice
    os.environ["SPEAKER_MODEL"] = model
    os.environ["SPEAKER_DISABLE_ASD"] = "1" if no_asd else "0"
    model_label, model_dir = MODEL_DISPLAY[model]

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    print(f"""
{CYAN}{BOLD}{'=' * 60}
   SPEAKER + FACE RECOGNITION SERVER
{'=' * 60}{RESET}

{BOLD}  Pipeline:{RESET} {('[BRAID-exclusive]' if braid else '[speaker+face]')}
    Face    InsightFace buffalo_l     {DIM}cuda:0{RESET}
    Voice   {model_label:<25}{DIM}cuda:1{RESET}
    Diar    DiariZen v2 (4-spk)       {DIM}cuda:2{RESET}
    BRAID   {'ENABLED (exclusive)' if braid else 'disabled':<25}{DIM}--braid{RESET}

{BOLD}  Storage:{RESET}
    Faces   /workspace/database/face_db/
    Voices  /workspace/database/embedding_accumulation_method/{model_dir}/
    BRAID   /workspace/database/braid_sys_db/

{DIM}  Loading models...{RESET}
""")

    if braid:
        # BRAID-exclusive mode: only BraidService is registered. The existing
        # SpeakerRecognitionService is *not* started in this mode to avoid
        # loading duplicate face/voice/diar models.
        from core_api.braid.grpc_handle import BraidServiceServicer
        pb2_grpc.add_BraidServiceServicer_to_server(
            BraidServiceServicer(),
            server,
        )
        logger.info(f"{GREEN}BRAID-exclusive mode: only BraidService registered "
                    f"(SpeakerRecognitionService disabled).{RESET}")
    else:
        from speaker_service import SpeakerRecognitionManager
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
    parser.add_argument("--braid", action="store_true",
                        help="Also register BraidService (BRAID closed-loop "
                             "perception/decision/action pipeline).")
    parser.add_argument("--no-asd", action="store_true",
                        help="Disable Active Speaker Detection face-bbox overlay "
                             "(applies to both ProcessVideo and RecognizeSpeakers).")
    args = parser.parse_args()
    serve(port=args.port, model=args.model, braid=args.braid, no_asd=args.no_asd)
