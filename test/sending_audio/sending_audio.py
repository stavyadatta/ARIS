import io
import sys
import grpc 
import time
import wave
from google.protobuf.empty_pb2 import Empty

from grpc_communication.grpc_pb2 import AudioRequest
from grpc_communication.grpc_pb2_grpc import MediaServiceStub


def sending_audio(file_name, stub):
    wav_file = wave.open(file_name, 'rb')
    num_channels = wav_file.getnchannels()
    # Get the sample rate (number of samples per second)
    sample_rate = wav_file.getframerate()
    # Get the sample width in bytes (e.g., 2 bytes for 16-bit audio)
    sample_width = wav_file.getsampwidth()
    # Get the total number of audio frames
    num_frames = wav_file.getnframes()
    # Read all the audio frames as bytes
    audio_bytes = wav_file.readframes(num_frames)
    # Determine the encoding based on the sample width
    wav_file.close()
    
    encoding_map = {
        1: "PCM_8",
        2: "PCM_16",
        3: "PCM_24",
        4: "PCM_32"
    }
    encoding = encoding_map.get(sample_width, "PCM_16")  # Default to PCM_16 if unknown
    
    request = AudioRequest(
        audio_data=audio_bytes,
        sample_rate=sample_rate,
        num_channels=num_channels,  # Assuming mono audio
        encoding=encoding,
        description="Audio captured by Pepper"
    )

    response = stub.SendAudio(request)
    print("SendAudio response: {} - {}".format(response.status, response.message))


def receive_llm_response(stub):
    request = Empty()
    try:
        response_stream = stub.LLmResponse(request)
        print("Receiving streamed text chunks:")
        for chunk in response_stream:
            sys.stdout.write(chunk.text)
            sys.stdout.flush()
            if chunk.is_final:
                sys.stdout.write("\n[Final chunk received]\n")
                sys.stdout.flush()
                break  # Exit after receiving the final chunk
    except grpc.RpcError as e:
        print("gRPC error: {} - {}".format(e.code(), e.details()))


if __name__ == "__main__":
    channel = grpc.insecure_channel("172.27.72.27:50051")
    stub = MediaServiceStub(channel)
    sending_audio("/workspace/pepper_client/received_audio/good_sample.wav", stub)
    receive_llm_response(stub)

    
