"""
Standalone client for Face + Speaker Recognition.

Usage (MacBook):
    python speaker_client.py --test-mode --server YOUR_SERVER_IP:50051

This captures audio from your microphone + video from your webcam,
streams both to the server, and prints identification results.

The server handles all the heavy lifting (face recognition + voice recognition).
Your MacBook only captures and sends raw data.
"""
import io
import os
import sys
import time
import uuid
import grpc
import queue
import logging
import argparse
import numpy as np
from threading import Thread

# Auto-detect grpc_communication path (works on both server and MacBook)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_grpc_path = os.path.join(os.path.dirname(_script_dir), "grpc_communication")
sys.path.insert(0, _grpc_path)
import grpc_pb2
import grpc_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("speaker_client")

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

def _ts():
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S")


class SpeakerClient:
    """
    Standalone face + speaker recognition client.

    Captures audio + webcam, streams to server, prints results.
    Server runs face recognition + voice recognition and returns combined results.
    """

    def __init__(self, server_address="localhost:50051"):
        self.server_address = server_address
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.is_active = False
        self.sample_rate = 16000
        self._segments_to_send = queue.Queue()

        # gRPC
        self.channel = grpc.insecure_channel(server_address)
        self.stub = grpc_pb2_grpc.SpeakerRecognitionServiceStub(self.channel)

        # Latest webcam frame (updated by camera thread)
        self._latest_frame_jpeg = None
        self._latest_frame_w = 0
        self._latest_frame_h = 0

        logger.info(f"  {DIM}{_ts()}{RESET}  {CYAN}>> CLIENT{RESET}     server={server_address}  session={self.session_id}")

    def _generate_segments(self):
        """Generator yielding SpeakerAudioSegment for gRPC stream."""
        while self.is_active:
            try:
                segment = self._segments_to_send.get(timeout=0.1)
                yield segment
            except queue.Empty:
                continue

    def _handle_responses(self):
        """Process and display server results."""
        try:
            responses = self.stub.RecognizeSpeakers(self._generate_segments())
            for result in responses:
                speaker = result.speaker_id or "unknown"
                conf = result.confidence

                if result.is_correction:
                    icon, label, color = "~~", "CORRECTION", YELLOW
                elif result.is_new_speaker:
                    icon, label, color = "**", "NEW VOICE", MAGENTA
                elif conf >= 0.6:
                    icon, label, color = "++", "RECOGNIZED", GREEN
                else:
                    icon, label, color = "??", "UNKNOWN", RED

                logger.info(
                    f"  {DIM}{_ts()}{RESET}  {color}{icon} {label:<14}{RESET} "
                    f"{BOLD}{speaker:<12}{RESET}  "
                    f"conf={conf:.4f}  t={result.segment_start_time:.1f}s  "
                    f"dur={result.segment_duration:.1f}s  "
                    f"{DIM}{result.status}{RESET}"
                )
        except grpc.RpcError as e:
            logger.error(f"  {DIM}{_ts()}{RESET}  {RED}!! gRPC ERROR{RESET}    {e.code()} - {e.details()}")
        except Exception as e:
            logger.error(f"  {DIM}{_ts()}{RESET}  {RED}!! ERROR{RESET}         {e}")

    def _audio_loop(self, energy_threshold=500):
        """Capture audio from mic (runs in BACKGROUND thread)."""
        try:
            import pyaudio
        except ImportError:
            logger.error("PyAudio not installed. Run: pip install pyaudio")
            return

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )

        conversation_start = time.time()
        segment_buffer = io.BytesIO()
        is_in_speech = False
        silence_frames = 0
        speech_start_time = 0.0
        max_silence_frames = 8

        try:
            while self.is_active:
                data = stream.read(1024, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
                current_time = time.time() - conversation_start

                if rms > energy_threshold:
                    silence_frames = 0
                    if not is_in_speech:
                        is_in_speech = True
                        segment_buffer = io.BytesIO()
                        speech_start_time = current_time
                        logger.info(f"  {DIM}{_ts()}{RESET}  {CYAN}<< SPEECH{RESET}      started at {current_time:.1f}s  {DIM}RMS={rms:.0f}{RESET}")
                    segment_buffer.write(data)
                else:
                    if is_in_speech:
                        segment_buffer.write(data)
                        silence_frames += 1
                        if silence_frames >= max_silence_frames:
                            is_in_speech = False
                            silence_frames = 0
                            segment_buffer.seek(0)
                            audio_bytes = segment_buffer.read()
                            duration = (len(audio_bytes) // 2) / self.sample_rate
                            if duration >= 0.5:
                                logger.info(f"  {DIM}{_ts()}{RESET}  {CYAN}>> SENDING{RESET}     {duration:.2f}s audio")
                                self._send_segment(audio_bytes, speech_start_time, duration)
        except Exception as e:
            logger.error(f"Audio loop error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    def _send_segment(self, audio_bytes, start_time, duration):
        """Queue an audio segment + latest camera frame for sending."""
        segment = grpc_pb2.SpeakerAudioSegment(
            audio_data=audio_bytes,
            sample_rate=self.sample_rate,
            num_channels=1,
            audio_encoding="PCM_16",
            segment_start_time=start_time,
            segment_duration=duration,
            session_id=self.session_id,
            face_id="",  # server determines face_id from image
            image_data=self._latest_frame_jpeg or b"",
            image_width=self._latest_frame_w,
            image_height=self._latest_frame_h
        )
        self._segments_to_send.put(segment)
        has_image = f"{GREEN}+cam{RESET}" if self._latest_frame_jpeg else f"{DIM}-cam{RESET}"
        logger.info(f"  {DIM}{_ts()}{RESET}  {BLUE}-> QUEUED{RESET}      {duration:.2f}s audio  {has_image}")

    def run_test_mode(self, use_camera=True):
        """
        Record from host mic + webcam.
        macOS requires cv2.imshow on the main thread, so:
        - Camera runs on MAIN thread (with imshow)
        - Audio + gRPC run on BACKGROUND threads
        Press Ctrl+C or 'q' in camera window to stop.
        """
        ENERGY_THRESHOLD = 500
        self.is_active = True

        cam_status = f"{GREEN}ON{RESET}" if use_camera else f"{RED}OFF{RESET}"
        print(f"""
{CYAN}{BOLD}{'=' * 55}
   SPEAKER + FACE RECOGNITION CLIENT
{'=' * 55}{RESET}

{BOLD}  Connection:{RESET}
    Server    {self.server_address}
    Session   {self.session_id}

{BOLD}  Capture:{RESET}
    Camera    {cam_status}
    Mic       {GREEN}ON{RESET}  {DIM}(threshold: {ENERGY_THRESHOLD} RMS){RESET}

{DIM}  Speak into mic. Press Ctrl+C to stop.{RESET}
{DIM}  Press 'q' in camera window to stop.{RESET}
{CYAN}{'=' * 55}{RESET}
""")

        # Background threads: gRPC responses + audio capture
        Thread(target=self._handle_responses, daemon=True).start()
        Thread(target=self._audio_loop, args=(ENERGY_THRESHOLD,), daemon=True).start()

        if use_camera:
            # Camera on MAIN thread (macOS requires this for cv2.imshow)
            try:
                import cv2
            except ImportError:
                logger.warning("OpenCV not available — running audio only")
                use_camera = False

        if use_camera:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.warning("Could not open webcam — running audio only")
                use_camera = False

        if use_camera:
            logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}OK WEBCAM{RESET}      capturing frames")
            try:
                while self.is_active:
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.01)
                        continue

                    h, w = frame.shape[:2]
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    self._latest_frame_jpeg = jpeg.tobytes()
                    self._latest_frame_w = w
                    self._latest_frame_h = h

                    cv2.imshow("Speaker Client - Camera", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info(f"\n  {YELLOW}Camera closed (q pressed){RESET}")
                        break
            except KeyboardInterrupt:
                logger.info(f"\n  {YELLOW}Stopping...{RESET}")
            finally:
                cap.release()
                cv2.destroyAllWindows()
        else:
            # No camera — just wait on main thread
            try:
                while self.is_active:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info(f"\n  {YELLOW}Stopping...{RESET}")

        self.is_active = False
        print(f"\n  {GREEN}Client finished.{RESET}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Face + Speaker Recognition Client (MacBook)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full mode (mic + webcam):
    python speaker_client.py --test-mode --server 192.168.1.100:50051

  Audio only (no webcam):
    python speaker_client.py --test-mode --no-camera --server 192.168.1.100:50051
        """
    )
    parser.add_argument("--server", type=str, default="localhost:50051",
                        help="Server address (default: localhost:50051)")
    parser.add_argument("--test-mode", action="store_true",
                        help="Use MacBook mic + webcam")
    parser.add_argument("--no-camera", action="store_true",
                        help="Disable webcam (audio only)")
    args = parser.parse_args()

    if args.test_mode:
        client = SpeakerClient(server_address=args.server)
        client.run_test_mode(use_camera=not args.no_camera)
    else:
        parser.print_help()
        print("\nRun with --test-mode for MacBook mic + webcam")
        sys.exit(1)


if __name__ == "__main__":
    main()
