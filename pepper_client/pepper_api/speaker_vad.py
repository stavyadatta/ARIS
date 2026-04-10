"""
Client-side VAD for speaker recognition streaming.

Runs alongside AudioManager2 (which handles the ProcessAudioImg flow).
This module independently detects speech segments and streams them
to the SpeakerRecognitionService via gRPC bidirectional streaming.

Does NOT modify AudioManager2 or the existing audio pipeline.
"""
import io
import queue
import time
import uuid
import grpc
import numpy as np
from threading import Thread

import sys
sys.path.insert(0, "/workspace/grpc_communication")
import grpc_pb2
import grpc_pb2_grpc


class SpeakerVAD:
    """
    Voice Activity Detection for speaker recognition streaming.

    Subscribes to Pepper's audio service, detects speech segments,
    and streams them to the server's SpeakerRecognitionService.
    """

    def __init__(self, session, server_address="localhost:50051",
                 energy_threshold=370, sample_rate=16000,
                 min_segment_duration=0.5, max_silence_loops=15):
        self.module_name = "SpeakerVAD"
        self.session = session
        self.server_address = server_address
        self.energy_threshold = energy_threshold
        self.sample_rate = sample_rate
        self.min_segment_duration = min_segment_duration
        self.max_silence_loops = max_silence_loops

        self.is_active = False
        self.session_id = None

        # gRPC channel and stub
        self.channel = grpc.insecure_channel(server_address)
        self.stub = grpc_pb2_grpc.SpeakerRecognitionServiceStub(self.channel)

        # Audio buffer for current speech segment
        self._segment_buffer = io.BytesIO()
        self._segment_start_time = 0.0
        self._is_in_speech = False
        self._silence_count = 0
        self._conversation_start_time = 0.0

        # Thread-safe queue for segments to send
        self._segments_to_send = queue.Queue()

    def _generate_segments(self):
        """
        Generator that yields SpeakerAudioSegment messages.
        Called by gRPC stub for the request stream.
        """
        while self.is_active:
            try:
                segment = self._segments_to_send.get(timeout=0.05)
                yield segment
            except queue.Empty:
                continue

    def start(self):
        """Start the VAD + speaker recognition stream."""
        self.is_active = True
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self._conversation_start_time = time.time()
        self._segments_to_send = queue.Queue()

        # Start the gRPC streaming call in a background thread
        self._response_thread = Thread(
            target=self._handle_responses,
            daemon=True
        )
        self._response_thread.start()

        # Subscribe to audio service
        audio_service = self.session.service("ALAudioDevice")
        audio_service.setClientPreferences(
            self.module_name, self.sample_rate, 1, 0  # mono
        )
        audio_service.subscribe(self.module_name)

    def stop(self):
        """Stop the VAD stream and clean up."""
        self._flush_segment()
        self.is_active = False

    def _handle_responses(self):
        """Process speaker identification results from the server."""
        try:
            responses = self.stub.RecognizeSpeakers(self._generate_segments())
            for result in responses:
                prefix = "[CORRECTION]" if result.is_correction else "[LIVE]"
                print(
                    f"{prefix} Speaker: {result.speaker_id}, "
                    f"Confidence: {result.confidence:.2f}, "
                    f"Time: {result.segment_start_time:.1f}s, "
                    f"New: {result.is_new_speaker}"
                )
        except grpc.RpcError as e:
            print(f"gRPC error in speaker recognition: {e}")

    def processRemote(self, nbOfChannels, nbOfSamplesByChannel, timeStamp, inputBuffer):
        """
        Called by Pepper's ALAudioDevice for each audio frame.
        Detects speech and accumulates into segments.
        """
        if not self.is_active:
            return

        audio_service = self.session.service("ALAudioDevice")
        current_energy = audio_service.getFrontMicEnergy()
        current_time = time.time() - self._conversation_start_time

        if current_energy > self.energy_threshold:
            # Speech detected
            self._silence_count = 0

            if not self._is_in_speech:
                self._is_in_speech = True
                self._segment_buffer = io.BytesIO()
                self._segment_start_time = current_time

            self._segment_buffer.write(inputBuffer)

        else:
            # Silence
            if self._is_in_speech:
                self._segment_buffer.write(inputBuffer)
                self._silence_count += 1

                if self._silence_count >= self.max_silence_loops:
                    self._flush_segment()

    def _flush_segment(self):
        """Send accumulated audio segment to server."""
        if not self._is_in_speech:
            return

        self._is_in_speech = False
        self._silence_count = 0

        self._segment_buffer.seek(0)
        audio_bytes = self._segment_buffer.read()

        # Check minimum duration
        num_samples = len(audio_bytes) // 2  # PCM_16
        duration = num_samples / self.sample_rate

        if duration < self.min_segment_duration:
            return

        segment = grpc_pb2.SpeakerAudioSegment(
            audio_data=audio_bytes,
            sample_rate=self.sample_rate,
            num_channels=1,
            audio_encoding="PCM_16",
            segment_start_time=self._segment_start_time,
            segment_duration=duration,
            session_id=self.session_id
        )

        self._segments_to_send.put(segment)
        self._segment_buffer = io.BytesIO()
