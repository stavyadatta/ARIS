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
        Bidirectional streaming RPC with integrated face recognition.

        For each segment:
        1. If image_data present → face recognition → face_id
        2. Otherwise use face_id from request (client-provided)
        3. SHORT PATH: immediate voice embedding + match + enroll
        4. LONG PATH: buffer audio → diarize at 10s → corrections
        """
        session_id = None
        current_face_id = None  # track across segments

        try:
            for request in request_iterator:
                audio_data = request.audio_data
                sample_rate = request.sample_rate or 16000
                session_id = request.session_id
                segment_start = request.segment_start_time
                segment_duration = request.segment_duration

                # === FACE RECOGNITION (if image provided) ===
                if request.image_data:
                    detected_face = self._extract_face_id(
                        request.image_data,
                        request.image_width,
                        request.image_height
                    )
                    if detected_face is not None:
                        current_face_id = detected_face

                # Fall back to client-provided face_id if no image
                if current_face_id is None and request.face_id:
                    current_face_id = request.face_id

                # === SHORT PATH: Immediate single-speaker result ===
                try:
                    min_bytes = int(0.5 * sample_rate * 2)
                    if len(audio_data) >= min_bytes:
                        result = self.speaker_recognition.identify_speaker(
                            audio_data, sample_rate,
                            current_face_id=current_face_id
                        )
                        yield pb2.SpeakerResult(
                            speaker_id=result["face_id"] or "",
                            confidence=result["confidence"],
                            segment_start_time=segment_start,
                            segment_duration=segment_duration,
                            is_new_speaker=result["is_new"],
                            session_id=session_id,
                            is_correction=False,
                            status=result["status"]
                        )
                except Exception as e:
                    _log("!!", "VOICE ERROR", str(e), _RED)
                    traceback.print_exc()

                # === LONG PATH: Buffer accumulation + batch diarization ===
                try:
                    diar_segments = self.diarization.accumulate_segment(
                        session_id=session_id,
                        audio_data=audio_data,
                        segment_start_time=segment_start,
                        sample_rate=sample_rate
                    )

                    if diar_segments is not None:
                        for seg in diar_segments:
                            seg_audio = seg["audio"]
                            seg_start = seg["start"]
                            seg_end = seg["end"]

                            min_seg_bytes = int(0.5 * sample_rate * 2)
                            if len(seg_audio) < min_seg_bytes:
                                continue

                            try:
                                result = self.speaker_recognition.identify_speaker(
                                    seg_audio, sample_rate,
                                    current_face_id=current_face_id
                                )
                                yield pb2.SpeakerResult(
                                    speaker_id=result["face_id"] or "",
                                    confidence=result["confidence"],
                                    segment_start_time=seg_start,
                                    segment_duration=seg_end - seg_start,
                                    is_new_speaker=result["is_new"],
                                    session_id=session_id,
                                    is_correction=True,
                                    status=result["status"]
                                )
                            except Exception as e:
                                _log("!!", "DIAR ERROR", str(e), _RED)
                                traceback.print_exc()

                except Exception as e:
                    _log("!!", "BUFFER ERROR", str(e), _RED)
                    traceback.print_exc()

        except Exception as e:
            _log("!!", "FATAL ERROR", str(e), _RED)
            traceback.print_exc()
        finally:
            if session_id is not None:
                self.diarization.clear_session(session_id)
