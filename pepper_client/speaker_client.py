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
        self._video_mode = False
        self._last_speaker = None  # track transitions for video mode

        logger.info(f"  {DIM}{_ts()}{RESET}  {CYAN}>> CLIENT{RESET}     server={server_address}  session={self.session_id}")

    def _generate_segments(self):
        """Generator yielding SpeakerAudioSegment for gRPC stream."""
        while self.is_active:
            try:
                segment = self._segments_to_send.get(timeout=0.1)
                yield segment
            except queue.Empty:
                continue

    def _format_video_time(self, seconds):
        """Format seconds as MM:SS for video timestamps."""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    def _handle_responses(self):
        """Process and display server results."""
        try:
            responses = self.stub.RecognizeSpeakers(self._generate_segments())
            for result in responses:
                speaker = result.speaker_id or "unknown"
                conf = result.confidence

                if self._video_mode:
                    # VIDEO MODE: only show transitions and new enrollments
                    vt = self._format_video_time(result.video_timestamp)

                    if result.is_new_speaker:
                        logger.info(
                            f"  {MAGENTA}{BOLD}  [{vt}]  NEW PERSON    "
                            f"{speaker}{RESET}  "
                            f"{DIM}(enrolled voice){RESET}"
                        )
                        self._last_speaker = speaker

                    elif speaker != self._last_speaker and speaker != "unknown":
                        # Speaker changed — this is a transition
                        if self._last_speaker:
                            logger.info(
                                f"  {CYAN}{BOLD}  [{vt}]  SWITCH        "
                                f"{self._last_speaker} -> {speaker}{RESET}  "
                                f"{DIM}conf={conf:.2f}{RESET}"
                            )
                        else:
                            logger.info(
                                f"  {GREEN}{BOLD}  [{vt}]  SPEAKING      "
                                f"{speaker}{RESET}  "
                                f"{DIM}conf={conf:.2f}{RESET}"
                            )
                        self._last_speaker = speaker

                    elif speaker == "unknown":
                        logger.info(
                            f"  {YELLOW}  [{vt}]  UNKNOWN       "
                            f"???{RESET}  "
                            f"{DIM}conf={conf:.2f}  {result.status}{RESET}"
                        )

                    elif result.is_correction:
                        logger.info(
                            f"  {DIM}  [{vt}]  correction    "
                            f"{speaker}  conf={conf:.2f}{RESET}"
                        )
                    # else: same speaker, no transition — stay quiet

                else:
                    # LIVE MODE: show everything
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

    def _send_segment(self, audio_bytes, start_time, duration, video_timestamp=0.0):
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
            image_height=self._latest_frame_h,
            video_timestamp=video_timestamp
        )
        self._segments_to_send.put(segment)
        has_image = f"{GREEN}+cam{RESET}" if self._latest_frame_jpeg else f"{DIM}-cam{RESET}"
        if not self._video_mode:
            logger.info(f"  {DIM}{_ts()}{RESET}  {BLUE}-> QUEUED{RESET}      {duration:.2f}s audio  {has_image}")

    def _video_audio_playback(self, audio_data, sample_rate):
        """Play extracted audio in background using PyAudio."""
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True,
                frames_per_buffer=4096
            )
            offset = 0
            chunk_size = 4096
            while offset < len(audio_data) and self.is_active:
                chunk = audio_data[offset:offset + chunk_size]
                if chunk:
                    stream.write(chunk)
                offset += chunk_size
            stream.stop_stream()
            stream.close()
            pa.terminate()
        except Exception as e:
            logger.warning(f"  {DIM}{_ts()}{RESET}  {YELLOW}!! AUDIO{RESET}       Playback failed: {e} (video will play without sound)")

    def _video_vad_sender(self, all_audio, sample_rate):
        """Run VAD on extracted audio and send segments to server (background thread)."""
        ENERGY_THRESHOLD = 500
        CHUNK_SAMPLES = 1024
        CHUNK_BYTES = CHUNK_SAMPLES * 2
        max_silence_frames = 8

        segment_buffer = io.BytesIO()
        is_in_speech = False
        silence_frames = 0
        speech_start_time = 0.0
        offset = 0

        while offset < len(all_audio) and self.is_active:
            chunk = all_audio[offset:offset + CHUNK_BYTES]
            offset += CHUNK_BYTES

            if len(chunk) < CHUNK_BYTES:
                break

            audio_np = np.frombuffer(chunk, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
            current_time = (offset // 2) / sample_rate

            if rms > ENERGY_THRESHOLD:
                silence_frames = 0
                if not is_in_speech:
                    is_in_speech = True
                    segment_buffer = io.BytesIO()
                    speech_start_time = current_time
                    logger.info(f"  {DIM}{_ts()}{RESET}  {CYAN}<< SPEECH{RESET}      at {current_time:.1f}s  {DIM}RMS={rms:.0f}{RESET}")
                segment_buffer.write(chunk)
            else:
                if is_in_speech:
                    segment_buffer.write(chunk)
                    silence_frames += 1
                    if silence_frames >= max_silence_frames:
                        is_in_speech = False
                        silence_frames = 0
                        segment_buffer.seek(0)
                        audio_bytes = segment_buffer.read()
                        duration = (len(audio_bytes) // 2) / sample_rate
                        if duration >= 0.5:
                            vt = self._format_video_time(speech_start_time)
                            logger.info(f"  {DIM}  [{vt}]  analyzing     {duration:.2f}s audio...{RESET}")
                            self._send_segment(audio_bytes, speech_start_time, duration, video_timestamp=speech_start_time)

        # Flush remaining
        if is_in_speech:
            segment_buffer.seek(0)
            audio_bytes = segment_buffer.read()
            duration = (len(audio_bytes) // 2) / sample_rate
            if duration >= 0.5:
                vt = self._format_video_time(speech_start_time)
                logger.info(f"  {DIM}  [{vt}]  analyzing     {duration:.2f}s audio (final)...{RESET}")
                self._send_segment(audio_bytes, speech_start_time, duration, video_timestamp=speech_start_time)

    def run_video_mode(self, video_path):
        """
        Play a video file with audio while running face + voice recognition.
        Video plays in an OpenCV window (main thread, macOS compatible).
        Audio plays through speakers via PyAudio (background thread).
        VAD + gRPC streaming runs in background threads.
        """
        import cv2
        import subprocess
        import tempfile
        import wave

        if not os.path.exists(video_path):
            logger.error(f"  {DIM}{_ts()}{RESET}  {RED}!! ERROR{RESET}         Video not found: {video_path}")
            return

        # Step 1: Extract audio from video → 16kHz mono PCM_16 WAV
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav.close()
        logger.info(f"  {DIM}{_ts()}{RESET}  {CYAN}.. EXTRACT{RESET}     extracting audio from video...")

        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                temp_wav.name
            ], capture_output=True, check=True)
        except FileNotFoundError:
            logger.error(f"  {DIM}{_ts()}{RESET}  {RED}!! ERROR{RESET}         ffmpeg not found. Install: brew install ffmpeg")
            return
        except subprocess.CalledProcessError as e:
            logger.error(f"  {DIM}{_ts()}{RESET}  {RED}!! ERROR{RESET}         ffmpeg failed: {e.stderr.decode()}")
            return

        with wave.open(temp_wav.name, 'rb') as wf:
            extracted_sample_rate = wf.getframerate()
            all_audio = wf.readframes(wf.getnframes())
        os.remove(temp_wav.name)

        total_audio_duration = (len(all_audio) // 2) / extracted_sample_rate

        # Step 2: Open video
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / video_fps
        frame_delay = 1.0 / video_fps

        print(f"""
{CYAN}{BOLD}{'=' * 55}
   VIDEO MODE — Playing + Recognizing
{'=' * 55}{RESET}

{BOLD}  Source:{RESET}
    Video     {os.path.basename(video_path)}
    Duration  {video_duration:.1f}s  {DIM}({total_frames} frames @ {video_fps:.0f}fps){RESET}
    Audio     {total_audio_duration:.1f}s  {DIM}(16kHz mono){RESET}

{BOLD}  Connection:{RESET}
    Server    {self.server_address}
    Session   {self.session_id}

{DIM}  Playing video with audio. Press 'q' to stop.{RESET}
{CYAN}{'=' * 55}{RESET}
""")

        self.is_active = True
        self._video_mode = True

        # Background threads: gRPC responses, audio playback, VAD + sending
        Thread(target=self._handle_responses, daemon=True).start()
        Thread(target=self._video_audio_playback, args=(all_audio, extracted_sample_rate), daemon=True).start()
        Thread(target=self._video_vad_sender, args=(all_audio, extracted_sample_rate), daemon=True).start()

        # Main thread: play video frames in sync with real-time clock (macOS needs imshow on main)
        playback_start = time.time()
        frame_num = 0

        try:
            while cap.isOpened() and self.is_active:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1
                video_time = frame_num / video_fps

                # Update latest frame for gRPC sending
                h, w = frame.shape[:2]
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                self._latest_frame_jpeg = jpeg.tobytes()
                self._latest_frame_w = w
                self._latest_frame_h = h

                # Draw timestamp overlay on the displayed frame
                timestamp_text = f"{video_time:.1f}s / {video_duration:.1f}s"
                cv2.putText(frame, timestamp_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow("Video - Speaker Recognition", frame)

                # Sync to real-time: wait until it's time to show the next frame
                elapsed = time.time() - playback_start
                wait_time = video_time - elapsed
                # cv2.waitKey needs at least 1ms
                wait_ms = max(1, int(wait_time * 1000))
                if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                    logger.info(f"\n  {YELLOW}Video stopped (q pressed){RESET}")
                    break

        except KeyboardInterrupt:
            logger.info(f"\n  {YELLOW}Stopped.{RESET}")

        # Wait briefly for remaining server responses
        logger.info(f"  {DIM}{_ts()}{RESET}  {DIM}.. WAITING{RESET}      waiting for final server responses...")
        time.sleep(3)

        cap.release()
        cv2.destroyAllWindows()
        self.is_active = False
        print(f"\n  {GREEN}Video processing finished.{RESET}\n")

    def run_process_video(self, video_path, max_duration=0.0):
        """
        Batch video processing mode:
        1. Upload video to server
        2. Server runs face + speaker recognition, annotates frames
        3. Download annotated video and save locally
        """
        if not os.path.exists(video_path):
            logger.error(f"  {RED}!! Video not found: {video_path}{RESET}")
            return

        file_size = os.path.getsize(video_path) / (1024 * 1024)
        filename = os.path.basename(video_path)

        dur_text = f"{max_duration:.0f}s" if max_duration > 0 else "full"
        print(f"""
{CYAN}{BOLD}{'=' * 55}
   BATCH VIDEO PROCESSING
{'=' * 55}{RESET}

{BOLD}  Input:{RESET}
    File      {filename}  {DIM}({file_size:.1f}MB){RESET}
    Duration  {dur_text}

{BOLD}  Connection:{RESET}
    Server    {self.server_address}

{DIM}  Uploading video to server for processing...{RESET}
{CYAN}{'=' * 55}{RESET}
""")

        # Upload video in chunks
        CHUNK_SIZE = 1024 * 1024  # 1MB

        def upload_chunks():
            with open(video_path, 'rb') as f:
                first = True
                while True:
                    data = f.read(CHUNK_SIZE)
                    if not data:
                        break
                    yield grpc_pb2.VideoUploadChunk(
                        data=data,
                        filename=filename if first else "",
                        max_duration_seconds=max_duration if first else 0.0
                    )
                    first = False

        logger.info(f"  {DIM}{_ts()}{RESET}  {CYAN}>> UPLOAD{RESET}      {filename} ({file_size:.1f}MB)")

        try:
            responses = self.stub.ProcessVideo(upload_chunks())

            output_file = None
            output_buf = bytearray()

            for chunk in responses:
                # Progress updates
                if chunk.progress and not chunk.data:
                    logger.info(f"  {DIM}{_ts()}{RESET}  {DIM}.. SERVER{RESET}      {chunk.progress}")
                    continue

                # First data chunk has the filename
                if chunk.filename:
                    output_file = chunk.filename
                    logger.info(f"  {DIM}{_ts()}{RESET}  {CYAN}<< DOWNLOAD{RESET}    {output_file}")

                if chunk.data:
                    output_buf.extend(chunk.data)

                if chunk.is_last:
                    break

            if output_buf and output_file:
                # Save to same directory as input
                output_dir = os.path.dirname(os.path.abspath(video_path))
                output_path = os.path.join(output_dir, output_file)
                with open(output_path, 'wb') as f:
                    f.write(output_buf)

                out_size = len(output_buf) / (1024 * 1024)
                print(f"""
{GREEN}{BOLD}{'=' * 55}
   DONE
{'=' * 55}{RESET}

  Saved: {BOLD}{output_path}{RESET}
  Size:  {out_size:.1f}MB

{DIM}  Open the video to see annotated face + speaker labels.{RESET}
{GREEN}{'=' * 55}{RESET}
""")
            else:
                logger.error(f"  {RED}!! No output received from server{RESET}")

        except grpc.RpcError as e:
            logger.error(f"  {RED}!! gRPC error: {e.code()} - {e.details()}{RESET}")
        except Exception as e:
            logger.error(f"  {RED}!! Error: {e}{RESET}")

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


def _parse_duration(s):
    """Parse duration string like '10m', '5m30s', '300' into seconds."""
    import re
    s = s.strip()
    # Pure number = seconds
    try:
        return float(s)
    except ValueError:
        pass
    # Pattern: 10m, 5m30s, 1h10m, etc.
    total = 0
    for val, unit in re.findall(r'(\d+)\s*(h|m|s)', s):
        val = int(val)
        if unit == 'h':
            total += val * 3600
        elif unit == 'm':
            total += val * 60
        elif unit == 's':
            total += val
    return float(total) if total > 0 else float(s)


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

  Stream video (real-time playback + recognition):
    python speaker_client.py --video /path/to/meeting.mp4 --server 192.168.1.100:50051

  Batch process video (server annotates, sends back labeled video):
    python speaker_client.py --process-video /path/to/meeting.mp4 --server 192.168.1.100:50051
    python speaker_client.py --process-video /path/to/meeting.mp4 --duration 10m --server IP:50051
    python speaker_client.py --process-video /path/to/meeting.mp4 --duration 300 --server IP:50051
        """
    )
    parser.add_argument("--server", type=str, default="localhost:50051",
                        help="Server address (default: localhost:50051)")
    parser.add_argument("--test-mode", action="store_true",
                        help="Use MacBook mic + webcam")
    parser.add_argument("--no-camera", action="store_true",
                        help="Disable webcam (audio only)")
    parser.add_argument("--video", type=str, default=None,
                        help="Stream video file (real-time playback + recognition)")
    parser.add_argument("--process-video", type=str, default=None,
                        help="Batch process: upload video, server annotates, download result")
    parser.add_argument("--duration", type=str, default=None,
                        help="Max duration to process (e.g., '10m', '5m30s', '300')")
    args = parser.parse_args()

    client = SpeakerClient(server_address=args.server)

    if args.process_video:
        duration = _parse_duration(args.duration) if args.duration else 0.0
        client.run_process_video(args.process_video, max_duration=duration)
    elif args.video:
        client.run_video_mode(args.video)
    elif args.test_mode:
        client.run_test_mode(use_camera=not args.no_camera)
    else:
        parser.print_help()
        print("\nRun with --test-mode, --video <path>, or --process-video <path>")
        sys.exit(1)


if __name__ == "__main__":
    main()
