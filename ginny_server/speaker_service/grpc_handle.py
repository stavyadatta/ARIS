import cv2
import logging
import datetime
import numpy as np
import traceback
import grpc_communication.grpc_pb2 as pb2
import grpc_communication.grpc_pb2_grpc as pb2_grpc

logger = logging.getLogger("speaker_recognition")

_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_MAGENTA = "\033[95m"
_BLUE = "\033[94m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

def _log(icon, label, detail, color=_RESET):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    logger.info(f"  {_DIM}{ts}{_RESET}  {color}{icon} {label:<14}{_RESET} {detail}")


class SpeakerRecognitionManager(pb2_grpc.SpeakerRecognitionServiceServicer):
    """
    gRPC servicer for combined Face + Speaker Recognition.

    For each incoming segment:
    1. If image_data is present → run face recognition → get face_id
    2. Run speaker recognition with that face_id context
    3. Return combined result (face_id + voice confidence + status)

    Dual-path audio:
    - Short path: immediate single-speaker embedding per segment
    - Long path: buffered diarization when >=10s accumulated
    """

    def __init__(self):
        super().__init__()
        from core_api import SpeakerRecognition, Diarization, FaceRecognition
        self.speaker_recognition = SpeakerRecognition
        self.diarization = Diarization
        self.face_recognition = FaceRecognition
        _log("OK", "SERVICER", "Face + Voice pipeline ready", _GREEN)

    def _extract_face_id(self, image_data, image_width, image_height):
        """
        Run face recognition on the provided image frame.
        Returns face_id (str) or None if no face detected.
        """
        if not image_data:
            return None

        try:
            # Decode JPEG bytes → numpy array
            img_array = np.frombuffer(image_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                logger.warning("Failed to decode image frame")
                return None

            # Use face recognition — recognize without enrolling first
            face_id, embedding = self.face_recognition.recognize_face_no_enroll(
                img, skip_validation=True
            )

            if face_id is not None:
                _log("^^", "FACE FOUND", f"{_BOLD}{face_id}{_RESET}", _BLUE)
                return face_id
            else:
                new_id = self.face_recognition.enroll_face(embedding, img)
                _log("**", "FACE NEW", f"{_BOLD}{new_id}{_RESET}  (enrolled)", _MAGENTA)
                return new_id

        except ValueError as e:
            _log("--", "NO FACE", str(e), _DIM)
            return None
        except Exception as e:
            _log("!!", "FACE ERROR", str(e), _RED)
            return None

    def RecognizeSpeakers(self, request_iterator, context):
        """
        Bidirectional streaming RPC with sliding-window diarization + multi-sample ReID.

        Audio accumulates in a per-session buffer. When buffer reaches WINDOW_SIZE (30s):
        1. Run DiariZen on the buffer → speaker clusters
        2. Concat audio per cluster → ERes2NetV2 embedding
        3. match_or_buffer against voice_db → voice_id or pending enrollment
        4. Yield SpeakerResult for each cluster's segments
        5. Shift buffer forward (keep WINDOW_OVERLAP for continuity)

        Face recognition runs immediately on each image frame (decoupled).
        """
        WINDOW_SIZE = 30.0      # seconds
        WINDOW_OVERLAP = 10.0   # seconds
        MIN_CLUSTER_AUDIO = 3.0 # seconds — skip shorter clusters
        MIN_ENROLL_SAMPLES = 3

        session_id = None
        current_face_id = None
        sample_rate = 16000

        # Per-session audio buffer
        audio_buffer = bytearray()
        buffer_start_time = 0.0  # absolute time of buffer start
        window_count = 0

        # Init enrollment buffer for this session
        self.speaker_recognition.init_enrollment_buffer()

        try:
            for request in request_iterator:
                audio_data = request.audio_data
                sample_rate = request.sample_rate or 16000
                session_id = request.session_id
                video_ts = request.video_timestamp

                # === FACE RECOGNITION (immediate, every frame) ===
                if request.image_data:
                    detected_face = self._extract_face_id(
                        request.image_data,
                        request.image_width,
                        request.image_height
                    )
                    if detected_face is not None:
                        current_face_id = detected_face

                if current_face_id is None and request.face_id:
                    current_face_id = request.face_id

                # === ACCUMULATE AUDIO ===
                audio_buffer.extend(audio_data)
                buffer_duration = (len(audio_buffer) // 2) / sample_rate

                # === WINDOW READY? Run diarization + ReID ===
                if buffer_duration >= WINDOW_SIZE:
                    window_count += 1
                    window_audio = bytes(audio_buffer)
                    win_start = buffer_start_time
                    win_end = win_start + buffer_duration

                    _log(">>", f"WINDOW {window_count}",
                         f"[{win_start:.0f}s - {win_end:.0f}s]  {buffer_duration:.1f}s audio",
                         _CYAN)

                    # Run diarization on the window
                    try:
                        diar_segments = self.diarization.diarize(window_audio, sample_rate)
                    except Exception as e:
                        _log("!!", "DIAR ERROR", str(e), _RED)
                        # Shift buffer and continue
                        shift_bytes = int(WINDOW_SIZE - WINDOW_OVERLAP) * sample_rate * 2
                        audio_buffer = audio_buffer[shift_bytes:]
                        buffer_start_time += (WINDOW_SIZE - WINDOW_OVERLAP)
                        continue

                    # Group diarized segments by speaker label
                    clusters = {}
                    for seg in diar_segments:
                        label = seg["speaker"]
                        if label not in clusters:
                            clusters[label] = {"audio": bytearray(), "segments": []}
                        clusters[label]["audio"].extend(seg["audio"])
                        clusters[label]["segments"].append(
                            (seg["start"] + win_start, seg["end"] + win_start)
                        )

                    _log("..", "CLUSTERS",
                         f"{len(clusters)} speakers in window {window_count}",
                         _GREEN)

                    # For each cluster: ERes2NetV2 embedding → match/buffer
                    for diar_label, cluster in clusters.items():
                        cluster_audio = bytes(cluster["audio"])
                        cluster_dur = (len(cluster_audio) // 2) / sample_rate

                        if cluster_dur < MIN_CLUSTER_AUDIO:
                            _log("--", "SKIP",
                                 f"{diar_label}: {cluster_dur:.1f}s (too short)", _DIM)
                            continue

                        try:
                            embedding = self.speaker_recognition.extract_embedding(
                                cluster_audio, sample_rate
                            )
                            result = self.speaker_recognition.match_or_buffer(
                                embedding, cluster_audio, sample_rate,
                                min_samples=MIN_ENROLL_SAMPLES
                            )

                            voice_id = result["voice_id"] or result.get("pending_id", "")
                            confidence = result["confidence"]

                            # Yield a result for each segment in this cluster
                            for seg_start, seg_end in cluster["segments"]:
                                yield pb2.SpeakerResult(
                                    speaker_id=voice_id,
                                    confidence=confidence,
                                    segment_start_time=seg_start,
                                    segment_duration=seg_end - seg_start,
                                    is_new_speaker=result["is_new"],
                                    session_id=session_id,
                                    is_correction=False,
                                    status=result["status"],
                                    video_timestamp=video_ts
                                )

                            _log("..", diar_label,
                                 f"{cluster_dur:.1f}s → {_BOLD}{voice_id}{_RESET}  "
                                 f"conf={confidence:.2f}  {_DIM}{result['status']}{_RESET}",
                                 _GREEN)

                        except Exception as e:
                            _log("!!", "CLUSTER ERR", f"{diar_label}: {e}", _RED)

                    # Shift buffer: keep last WINDOW_OVERLAP seconds
                    shift_bytes = int((WINDOW_SIZE - WINDOW_OVERLAP) * sample_rate * 2)
                    audio_buffer = audio_buffer[shift_bytes:]
                    buffer_start_time += (WINDOW_SIZE - WINDOW_OVERLAP)

        except Exception as e:
            _log("!!", "FATAL ERROR", str(e), _RED)
            traceback.print_exc()
        finally:
            # Flush pending enrollments at end of session
            if hasattr(self.speaker_recognition, '_pending'):
                flushed = self.speaker_recognition.flush_all_pending(sample_rate, min_samples=2)
                for r in flushed:
                    _log("**", "FLUSH", f"Enrolled {r['voice_id']} at session end", _GREEN)

            if session_id is not None:
                self.diarization.clear_session(session_id)
            _log("--", "SESSION END", f"{session_id}  windows={window_count}", _DIM)

    def ProcessVideo(self, request_iterator, context):
        """
        Batch video processing RPC.
        Client uploads video → server annotates with face+speaker labels → sends back.
        """
        import os
        import tempfile
        from speaker_service.video_processor import process_video

        # Step 1: Receive video file chunks
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        max_duration = 0.0
        filename = "input.mp4"

        for chunk in request_iterator:
            if chunk.filename:
                filename = chunk.filename
            if chunk.max_duration_seconds > 0:
                max_duration = chunk.max_duration_seconds
            temp_input.write(chunk.data)

        temp_input.close()
        _log("<<", "VIDEO RECV", f"{filename} ({os.path.getsize(temp_input.name) / 1024 / 1024:.1f}MB), max_dur={max_duration}s", _CYAN)

        # Step 2: Process video
        output_path = None
        try:
            for progress, result_path in process_video(
                temp_input.name, max_duration,
                self.face_recognition, self.speaker_recognition, self.diarization
            ):
                if progress:
                    yield pb2.VideoDownloadChunk(
                        data=b"", filename="", is_last=False, progress=progress
                    )
                if result_path:
                    output_path = result_path
        except Exception as e:
            _log("!!", "VIDEO ERROR", str(e), _RED)
            traceback.print_exc()
            os.remove(temp_input.name)
            return

        os.remove(temp_input.name)

        if not output_path or not os.path.exists(output_path):
            _log("!!", "VIDEO ERROR", "No output produced", _RED)
            return

        # Step 3: Stream annotated video back to client
        out_filename = f"annotated_{filename}"
        file_size = os.path.getsize(output_path)
        _log(">>", "VIDEO SEND", f"{out_filename} ({file_size / 1024 / 1024:.1f}MB)", _GREEN)

        CHUNK_SIZE = 1024 * 1024  # 1MB chunks
        with open(output_path, 'rb') as f:
            first = True
            while True:
                data = f.read(CHUNK_SIZE)
                if not data:
                    break
                yield pb2.VideoDownloadChunk(
                    data=data,
                    filename=out_filename if first else "",
                    is_last=False,
                    progress=""
                )
                first = False

        yield pb2.VideoDownloadChunk(data=b"", filename="", is_last=True, progress="Done")
        os.remove(output_path)
        _log("OK", "VIDEO DONE", f"Sent {out_filename} to client", _GREEN)
