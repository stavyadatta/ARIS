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

    def __init__(self, server_address="localhost:50051",
                 max_retries=5, connect_timeout=5.0,
                 initial_backoff=1.0, max_backoff=30.0,
                 stream_retry_backoff=2.0, max_stream_retries=10):
        self.server_address = server_address
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.is_active = False
        self.sample_rate = 16000
        self._segments_to_send = queue.Queue()
        self.max_retries = max_retries
        self.connect_timeout = connect_timeout
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.stream_retry_backoff = stream_retry_backoff
        self.max_stream_retries = max_stream_retries
        self._shutdown_called = False

        # gRPC — connect with retry
        self.channel = grpc.insecure_channel(server_address)
        self._connect_with_retry()
        self.stub = grpc_pb2_grpc.SpeakerRecognitionServiceStub(self.channel)

        # Latest webcam frame (updated by camera thread)
        self._latest_frame_jpeg = None
        self._latest_frame_w = 0
        self._latest_frame_h = 0
        self._video_mode = False
        self._last_speaker = None  # track transitions for video mode

        logger.info(f"  {DIM}{_ts()}{RESET}  {CYAN}>> CLIENT{RESET}     server={server_address}  session={self.session_id}")

    def _connect_with_retry(self):
        """Wait for channel to be ready with exponential backoff."""
        backoff = self.initial_backoff
        for attempt in range(1 + self.max_retries):
            try:
                future = grpc.channel_ready_future(self.channel)
                future.result(timeout=self.connect_timeout)
                logger.info(
                    f"  {DIM}{_ts()}{RESET}  {GREEN}CONNECTED{RESET}    "
                    f"to {self.server_address} (attempt {attempt + 1})"
                )
                return
            except grpc.FutureTimeoutError:
                if attempt < self.max_retries:
                    logger.warning(
                        f"  {DIM}{_ts()}{RESET}  {YELLOW}RETRY{RESET}        "
                        f"attempt {attempt + 1}/{self.max_retries}  "
                        f"backoff={backoff:.1f}s"
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, self.max_backoff)
                else:
                    logger.error(
                        f"  {DIM}{_ts()}{RESET}  {RED}FAILED{RESET}       "
                        f"could not connect to {self.server_address} "
                        f"after {self.max_retries} retries"
                    )
                    raise ConnectionError(
                        f"Could not connect to {self.server_address} "
                        f"after {self.max_retries} retries"
                    )

    def shutdown(self):
        """Gracefully shut down the client."""
        if self._shutdown_called:
            return
        self._shutdown_called = True
        self.is_active = False
        # Drain the queue
        while not self._segments_to_send.empty():
            try:
                self._segments_to_send.get_nowait()
            except queue.Empty:
                break
        self.channel.close()

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

    def run_pepper_mode(self, robot_ip, robot_port=9559, movement_cooldown=5.0,
                        listen_ip="192.168.0.50", listen_port=9559,
                        localizer="naoqi"):
        """
        Run on Pepper robot with:
        - Audio from Pepper's mic (16kHz mono via ALAudioDevice + processRemote callback)
        - Camera from Pepper's top camera (1280x960 RGB via ALVideoDevice)
        - Sound localisation → body rotation toward speaker
        - Speaker recognition via gRPC to server (audio + camera frames)

        localizer: "naoqi" (ALSoundLocalization), "srp-phat", or "srp-hsda"
        """
        try:
            import qi
            import cv2
            from PIL import Image as PILImage
        except ImportError as e:
            logger.error(f"  {RED}!! Missing dependency: {e}. Need qi + opencv + Pillow.{RESET}")
            return

        localizer_labels = {
            "naoqi": "ALSoundLocalization (NAOqi)",
            "srp-phat": "SRP-PHAT (4-mic, 48kHz)",
            "srp-hsda": "SRP-PHAT-HSDA (4-mic, 48kHz)",
        }
        loc_label = localizer_labels.get(localizer, localizer)

        connection_url = f"tcp://{robot_ip}:{robot_port}"
        print(f"""
{CYAN}{BOLD}{'=' * 55}
   PEPPER ROBOT MODE
{'=' * 55}{RESET}

{BOLD}  Robot:{RESET}
    Address   {connection_url}

{BOLD}  Pipeline:{RESET}
    Audio     Pepper mic (16kHz mono) → gRPC
    Camera    Pepper top cam (1280x960) → JPEG → gRPC
    Localise  {loc_label} → body rotation
    Recog     gRPC → {self.server_address}

{DIM}  Connecting to Pepper...{RESET}
{CYAN}{'=' * 55}{RESET}
""")

        app = qi.Application(["SpeakerClient", "--qi-url=" + connection_url])
        app.start()
        session = app.session
        logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}OK PEPPER{RESET}      Connected to {connection_url}")

        # Services
        motion = session.service("ALMotion")
        memory = session.service("ALMemory")
        audio_device = session.service("ALAudioDevice")
        sound_loc = session.service("ALSoundLocalization")
        video_service = session.service("ALVideoDevice")
        life_service = session.service("ALAutonomousLife")
        posture = session.service("ALRobotPosture")

        # Disable autonomous life (prevents Pepper from overriding camera/movement)
        life_service.setAutonomousAbilityEnabled("All", False)
        logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}OK LIFE{RESET}        Autonomous abilities disabled")

        # Wake robot and stand upright
        motion.wakeUp()
        posture.goToPosture("Stand", 0.5)
        logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}OK MOTION{RESET}      Robot awake, standing upright")

        # Enable sound localisation
        _srp_localizer = None
        _srp_audio_queue = None
        if localizer == "naoqi":
            sound_loc.subscribe("SpeakerClient")
            sound_loc.setParameter("Sensitivity", 0.8)
            audio_device.enableEnergyComputation()
            logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}OK AUDIO SVC{RESET}   ALSoundLocalization ready")
        else:
            # SRP-PHAT or SRP-PHAT-HSDA
            audio_device.enableEnergyComputation()
            import queue as _q
            _srp_audio_queue = _q.Queue(maxsize=10)

            from wake_word_localizer.srp_phat_localizer import SRPPHATLocalizer
            if localizer == "srp-phat":
                _srp_localizer = SRPPHATLocalizer(
                    sample_rate=48000, enable_msw=False, mic_acceptance_deg=180
                )
            else:  # srp-hsda
                _srp_localizer = SRPPHATLocalizer(
                    sample_rate=48000, enable_msw=True, mic_acceptance_deg=150
                )
            logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}OK AUDIO SVC{RESET}   {loc_label} ready")

        # Subscribe to camera (same settings as pepper.py: resolution=5, colorspace=11, fps=30)
        video_client = video_service.subscribeCamera(
            "SpeakerClientCam", 0, 5, 11, 30  # top cam, 1280x960, RGB, 30fps
        )
        video_service.setAllParametersToDefault(0)
        video_service.setParameter(0, 8, 1)  # Vertical flip (Pepper's cam is upside down)
        logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}OK CAMERA{RESET}      Top camera subscribed (1280x960 @ 30fps)")

        # ---- Audio capture via qi service (processRemote callback) ----
        # We need to register a qi service that receives audio buffers.
        # This is how AudioManager2 and SRPAudioService work.
        _audio_buffer = io.BytesIO()
        _audio_lock = __import__('threading').Lock()
        _is_speech = [False]
        _silence_count = [0]
        _speech_start = [0.0]
        _conv_start = time.time()
        _direction_samples = []
        _last_move_time = [0.0]
        _energy_threshold = 370
        _max_silence = 20
        _min_segment = 2.0
        _client_ref = self  # reference for the callback

        class PepperAudioCapture(object):
            """qi service that receives audio via processRemote and does VAD."""
            def __init__(self):
                self.module_name = "PepperAudioCapture"

            def init_service(self, sess):
                self.audio_service = sess.service("ALAudioDevice")

            def start(self):
                self.audio_service.setClientPreferences(
                    self.module_name, 16000, 1, 0  # 16kHz, mono
                )
                self.audio_service.subscribe(self.module_name)

            def stop(self):
                try:
                    self.audio_service.unsubscribe(self.module_name)
                except Exception:
                    pass

            def processRemote(self, nbOfChannels, nbOfSamplesByChannel, timeStamp, inputBuffer):
                if not _client_ref.is_active:
                    return

                current_energy = self.audio_service.getFrontMicEnergy()
                current_time = time.time() - _conv_start

                # Poll localisation during speech
                if _is_speech[0]:
                    if _srp_localizer is None:
                        # NAOqi localisation
                        try:
                            loc_data = memory.getData("ALSoundLocalization/SoundLocated")
                            if loc_data and len(loc_data) >= 2:
                                az = loc_data[1][0]
                                conf = loc_data[1][2]
                                if conf > 0.3:
                                    _direction_samples.append((az, conf))
                        except Exception:
                            pass

                if current_energy > _energy_threshold:
                    _silence_count[0] = 0
                    if not _is_speech[0]:
                        _is_speech[0] = True
                        with _audio_lock:
                            _audio_buffer.seek(0)
                            _audio_buffer.truncate(0)
                        _speech_start[0] = current_time
                        _direction_samples.clear()
                        logger.info(
                            f"  {DIM}{_ts()}{RESET}  {CYAN}<< SPEECH{RESET}      "
                            f"at {current_time:.1f}s  {DIM}energy={current_energy}{RESET}"
                        )

                    # Write audio bytes to buffer
                    with _audio_lock:
                        _audio_buffer.write(inputBuffer)

                else:
                    if _is_speech[0]:
                        # Still write during silence (capture trailing audio)
                        with _audio_lock:
                            _audio_buffer.write(inputBuffer)

                        _silence_count[0] += 1
                        if _silence_count[0] >= _max_silence:
                            _is_speech[0] = False
                            _silence_count[0] = 0
                            duration = current_time - _speech_start[0]

                            if duration >= _min_segment:
                                with _audio_lock:
                                    _audio_buffer.seek(0)
                                    audio_bytes = _audio_buffer.read()

                                logger.info(
                                    f"  {DIM}{_ts()}{RESET}  {CYAN}>> SENDING{RESET}     "
                                    f"{duration:.2f}s audio"
                                )

                                # Send to server via gRPC
                                _client_ref._send_segment(
                                    audio_bytes, _speech_start[0], duration
                                )

                                # Body movement from averaged direction
                                now = time.time()
                                if _direction_samples and (now - _last_move_time[0]) > movement_cooldown:
                                    total_conf = sum(c for _, c in _direction_samples)
                                    if total_conf > 0:
                                        avg_az = sum(az * c for az, c in _direction_samples) / total_conf
                                        avg_conf = total_conf / len(_direction_samples)
                                        logger.info(
                                            f"  {DIM}{_ts()}{RESET}  {MAGENTA}>> MOVE BODY{RESET}   "
                                            f"az={avg_az:.2f}rad ({avg_az * 57.3:.0f}°)  "
                                            f"conf={avg_conf:.2f}  "
                                            f"{DIM}({len(_direction_samples)} samples){RESET}"
                                        )
                                        try:
                                            motion.moveTo(0, 0, float(avg_az))
                                            _last_move_time[0] = time.time()
                                        except Exception as e:
                                            logger.error(f"  {DIM}{_ts()}{RESET}  {RED}!! MOVE{RESET}  {e}")

                                _direction_samples.clear()
                            else:
                                logger.info(
                                    f"  {DIM}{_ts()}{RESET}  {DIM}-- SKIP{RESET}        "
                                    f"{duration:.2f}s (need {_min_segment}s){RESET}"
                                )

        # Register and start audio capture service
        audio_capture = PepperAudioCapture()
        listen_url = f"tcp://{listen_ip}:{listen_port}"
        session.listen(listen_url)
        session.registerService("PepperAudioCapture", audio_capture)
        audio_capture.init_service(session)
        audio_capture.start()
        logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}OK AUDIO{RESET}       Audio capture registered, listening on {listen_url}")

        # Set active BEFORE starting SRP threads (they check is_active)
        self.is_active = True

        # ---- SRP-PHAT: 4-channel audio capture + localization thread ----
        _srp_audio_capture = None
        if _srp_localizer is not None:
            class SRPAudioCapture(object):
                """qi service capturing 4-channel 48kHz audio for SRP-PHAT."""
                def __init__(self):
                    self.module_name = "SRPAudioCapture"

                def init_service(self, sess):
                    self.audio_service = sess.service("ALAudioDevice")

                def start(self):
                    self.audio_service.setClientPreferences(
                        self.module_name, 48000, 0, 1  # 48kHz, all channels, deinterleaved
                    )
                    self.audio_service.subscribe(self.module_name)

                def stop(self):
                    try:
                        self.audio_service.unsubscribe(self.module_name)
                    except Exception:
                        pass

                def processRemote(self, nbOfChannels, nbOfSamplesByChannel, timeStamp, inputBuffer):
                    if not _client_ref.is_active:
                        return
                    try:
                        audio_np = np.frombuffer(inputBuffer, dtype=np.int16).astype(np.float32) / 32768.0
                        # Deinterleaved: reshape to (n_samples, n_channels)
                        audio_np = audio_np.reshape(nbOfChannels, nbOfSamplesByChannel).T
                        if _srp_audio_queue is not None and not _srp_audio_queue.full():
                            _srp_audio_queue.put_nowait(audio_np)
                    except Exception:
                        pass

            _srp_audio_capture = SRPAudioCapture()
            session.registerService("SRPAudioCapture", _srp_audio_capture)
            _srp_audio_capture.init_service(session)
            _srp_audio_capture.start()
            logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}OK SRP AUDIO{RESET}   4-channel capture at 48kHz")

            # SRP localization thread — runs SRP-PHAT on queued audio buffers
            def _srp_localization_loop():
                _srp_min_confidence = 1.5
                _srp_count = [0]
                while _client_ref.is_active:
                    try:
                        audio_chunk = _srp_audio_queue.get(timeout=0.2)
                        az, _, conf = _srp_localizer.locate(audio_chunk)
                        _srp_count[0] += 1
                        if conf >= _srp_min_confidence:
                            _direction_samples.append((az, conf))
                            if _srp_count[0] % 5 == 0:
                                logger.info(
                                    f"  {DIM}{_ts()}{RESET}  {MAGENTA}>> SRP-DOA{RESET}     "
                                    f"az={az:.2f}rad ({az * 57.3:.0f}°)  "
                                    f"conf={conf:.1f}"
                                )
                    except Exception as e:
                        if not isinstance(e, __import__('queue').Empty):
                            logger.error(
                                f"  {DIM}{_ts()}{RESET}  {RED}!! SRP ERR{RESET}     {e}"
                            )

            Thread(target=_srp_localization_loop, daemon=True).start()
            logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}OK SRP THREAD{RESET}  Localization thread running")

        # Start gRPC response handler
        Thread(target=self._handle_responses, daemon=True).start()

        # Camera capture thread
        def _pepper_camera_loop():
            while self.is_active:
                try:
                    raw = video_service.getImageRemote(video_client)
                    video_service.releaseImage(video_client)
                    if raw is None:
                        time.sleep(0.05)
                        continue
                    w, h = raw[0], raw[1]
                    pil_img = PILImage.frombytes("RGB", (w, h), bytes(raw[6]))
                    np_img = np.array(pil_img)
                    cv2_img = np_img[:, :, ::-1]  # RGB → BGR
                    cv2_img = cv2.flip(cv2_img, 0)  # Pepper cam is flipped
                    _, jpeg = cv2.imencode('.jpg', cv2_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    self._latest_frame_jpeg = jpeg.tobytes()
                    self._latest_frame_w = w
                    self._latest_frame_h = h
                except Exception as e:
                    logger.debug(f"Camera error: {e}")
                time.sleep(0.1)

        Thread(target=_pepper_camera_loop, daemon=True).start()
        logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}OK CAM THREAD{RESET}  Capturing frames in background")

        logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}{BOLD}READY{RESET}          Listening for speech...")

        # Main thread just waits
        try:
            while self.is_active:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info(f"\n  {YELLOW}Stopping...{RESET}")

        # ---- SHUTDOWN ----
        self.is_active = False
        logger.info(f"  {DIM}{_ts()}{RESET}  {YELLOW}.. SHUTDOWN{RESET}     Stopping Pepper services...")

        try:
            audio_capture.stop()
            logger.info(f"  {DIM}{_ts()}{RESET}  {DIM}   Audio unsubscribed{RESET}")
        except Exception:
            pass

        if _srp_audio_capture is not None:
            try:
                _srp_audio_capture.stop()
                logger.info(f"  {DIM}{_ts()}{RESET}  {DIM}   SRP audio unsubscribed{RESET}")
            except Exception:
                pass

        if localizer == "naoqi":
            try:
                sound_loc.unsubscribe("SpeakerClient")
                logger.info(f"  {DIM}{_ts()}{RESET}  {DIM}   Sound localisation unsubscribed{RESET}")
            except Exception:
                pass

        try:
            video_service.unsubscribe(video_client)
            logger.info(f"  {DIM}{_ts()}{RESET}  {DIM}   Camera unsubscribed{RESET}")
        except Exception:
            pass

        try:
            posture.goToPosture("Crouch", 0.5)
            logger.info(f"  {DIM}{_ts()}{RESET}  {DIM}   Robot crouching{RESET}")
        except Exception:
            pass

        try:
            life_service.setAutonomousAbilityEnabled("All", True)
            logger.info(f"  {DIM}{_ts()}{RESET}  {DIM}   Autonomous life re-enabled{RESET}")
        except Exception:
            pass

        try:
            motion.rest()
            logger.info(f"  {DIM}{_ts()}{RESET}  {GREEN}OK SHUTDOWN{RESET}    Robot is asleep (stiffness off)")
        except Exception as e:
            logger.error(f"  {DIM}{_ts()}{RESET}  {RED}!! REST ERROR{RESET}  {e}")

        print(f"\n  {GREEN}Pepper mode finished. Robot is resting.{RESET}\n")

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

  Pepper robot mode (audio + localisation + body movement):
    python speaker_client.py --robot-ip 192.168.0.52 --server 192.168.1.100:50051
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
    parser.add_argument("--robot-ip", type=str, default=None,
                        help="Pepper robot IP (enables robot mode with localisation + body movement)")
    parser.add_argument("--robot-port", type=int, default=9559,
                        help="Pepper NAOqi port (default: 9559)")
    parser.add_argument("--listen-ip", type=str, default="192.168.0.50",
                        help="IP for qi session listener on Pi (default: 192.168.0.50)")
    parser.add_argument("--listen-port", type=int, default=9559,
                        help="Port for qi session listener on Pi (default: 9559)")
    parser.add_argument("--localizer", choices=["naoqi", "srp-phat", "srp-hsda"],
                        default="naoqi",
                        help="Sound localisation method (default: naoqi)")
    args = parser.parse_args()

    client = SpeakerClient(server_address=args.server)

    if args.process_video:
        duration = _parse_duration(args.duration) if args.duration else 0.0
        client.run_process_video(args.process_video, max_duration=duration)
    elif args.video:
        client.run_video_mode(args.video)
    elif args.robot_ip:
        client.run_pepper_mode(args.robot_ip, args.robot_port,
                               listen_ip=args.listen_ip, listen_port=args.listen_port,
                               localizer=args.localizer)
    elif args.test_mode:
        client.run_test_mode(use_camera=not args.no_camera)
    else:
        parser.print_help()
        print("\nRun with --test-mode, --video <path>, --process-video <path>, or --robot-ip <IP>")
        sys.exit(1)


if __name__ == "__main__":
    main()
