"""Drive ProcessAudioImg with a fixed audio/image pair and report client-side latency.

Runs inside the iris-server container, where grpc and the generated stubs live.
Measures what the robot would feel: time to the first TextChunk, and total.
"""

import sys
import time
import wave

import cv2
import grpc

from grpc_pb2 import AudioImgRequest
from grpc_pb2_grpc import MediaServiceStub

SERVER_ADDRESS = "localhost:50051"
JPEG_EXTENSION = ".jpg"
MS_PER_SECOND = 1000


def load_pcm(wav_path):
    with wave.open(wav_path, "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        return frames, wav_file.getframerate(), wav_file.getnchannels()


def load_jpeg(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"could not read {image_path}")
    encoded, buffer = cv2.imencode(JPEG_EXTENSION, image)
    if not encoded:
        raise SystemExit(f"could not encode {image_path}")
    height, width = image.shape[:2]
    return buffer.tobytes(), width, height


def build_request(wav_path, image_path):
    audio_data, sample_rate, num_channels = load_pcm(wav_path)
    image_data, width, height = load_jpeg(image_path)
    return AudioImgRequest(
        audio_data=audio_data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        audio_encoding="PCM_16",
        audio_description="latency probe",
        image_data=image_data,
        image_format="JPEG",
        image_width=width,
        image_height=height,
        api_task="",
        skip_face_validation=True,
    )


def run_turn(stub, request, label):
    started_at = time.perf_counter()
    first_chunk_ms = None
    chunks = []

    for chunk in stub.ProcessAudioImg(request):
        if first_chunk_ms is None:
            first_chunk_ms = (time.perf_counter() - started_at) * MS_PER_SECOND
        chunks.append((chunk.mode, chunk.text))

    total_ms = (time.perf_counter() - started_at) * MS_PER_SECOND
    print(f"[client] {label} ttfa_ms={first_chunk_ms} total_ms={total_ms:.1f} "
          f"chunks={len(chunks)}")
    for mode, text in chunks:
        print(f"[client]   mode={mode!r} text={text[:160]!r}")
    return total_ms


def main():
    wav_path, image_path, label, turns = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    request = build_request(wav_path, image_path)
    stub = MediaServiceStub(grpc.insecure_channel(SERVER_ADDRESS))

    for turn_number in range(1, turns + 1):
        run_turn(stub, request, f"{label} turn={turn_number}/{turns}")


if __name__ == "__main__":
    main()
