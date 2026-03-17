import os
import glob
import wave
import torch
import logging
import tempfile
import numpy as np
import threading
from pathlib import Path
from typing import Tuple, Optional, List

from modelscope.models import Model
from sklearn.metrics.pairwise import cosine_similarity

# ANSI colors
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_MAGENTA = "\033[95m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

# Set up logger for speaker recognition
logger = logging.getLogger("speaker_recognition")
logger.setLevel(logging.INFO)

# File handler — logs to ginny_server/logs/speaker_recognition.log
_log_dir = Path(__file__).parent.parent.parent / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)
_file_handler = logging.FileHandler(_log_dir / "speaker_recognition.log")
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))
logger.addHandler(_file_handler)

# Stdout handler — clean colored output
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_stream_handler)


def _log_event(icon, label, detail, color=_RESET):
    """Pretty print a log event."""
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    logger.info(f"  {_DIM}{ts}{_RESET}  {color}{icon} {label:<14}{_RESET} {detail}")


class _SpeakerRecognition:
    """
    Speaker recognition using ERes2NetV2 (192-dim embeddings, 16kHz).

    Mirrors the _FaceRecognition pattern:
    - Loads voice embeddings from /workspace/database/voice_db/*.npy at startup
    - Matches via in-memory cosine similarity (sklearn)
    - Saves new voice embeddings as face_N.npy (keyed by face_id)
    - Thread-safe via Lock
    """

    def __init__(self,
                 model_id: str = "iic/speech_eres2netv2_sv_en_voxceleb_16k",
                 db_dir: str = "/workspace/database/voice_db",
                 recognition_threshold: float = 0.6,
                 device: str = "cuda:1"):
        self.device = device
        self.db_dir = Path(db_dir)
        self.recognition_threshold = recognition_threshold
        self._lock = threading.Lock()

        # Ensure voice_db directory exists
        self._ensure_db_directory()

        # Load ERes2NetV2 model directly for embedding extraction
        _log_event("...", "MODEL LOAD", f"ERes2NetV2 on {device}", _DIM)
        self.model = Model.from_pretrained(model_id, device=device)
        self.model.eval()
        _log_event("OK", "MODEL READY", f"ERes2NetV2 on {device}", _GREEN)

        # Load existing voice embeddings from disk
        self.known_ids, self.known_embeddings = self._load_database()
        _log_event("DB", "VOICE DB", f"{len(self.known_ids)} voices loaded from {self.db_dir}", _CYAN)

    def _ensure_db_directory(self):
        """Ensure that the voice database directory exists."""
        if not self.db_dir.exists():
            self.db_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created voice DB directory: {self.db_dir}")

    def _load_database(self) -> Tuple[List[str], np.ndarray]:
        """
        Load known voice embeddings and IDs from the database directory.
        Mirrors: face_recognition.py lines 160-182

        Returns:
            (list of face_ids, np.ndarray of embeddings stacked)
        """
        embedding_files = glob.glob(str(self.db_dir / "*.npy"))
        known_ids = []
        known_embeddings = []

        for ef in embedding_files:
            face_id = Path(ef).stem  # e.g., "face_1" from "face_1.npy"
            emb = np.load(ef)
            known_ids.append(face_id)
            known_embeddings.append(emb.flatten())

        if known_embeddings:
            known_embeddings = np.vstack(known_embeddings)
        else:
            known_embeddings = np.array([])

        return known_ids, known_embeddings

    def _audio_bytes_to_temp_wav(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Save raw PCM_16 bytes to a temporary WAV file. Returns path."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file_path = temp_file.name
        temp_file.close()
        try:
            with wave.open(temp_file_path, 'wb') as wave_file:
                wave_file.setnchannels(1)
                wave_file.setsampwidth(2)
                wave_file.setframerate(sample_rate)
                wave_file.writeframes(audio_data)
            return temp_file_path
        except Exception:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise

    def extract_embedding(self, audio_data: bytes, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract 192-dim embedding from raw PCM_16 audio bytes.
        Thread-safe via Lock.

        Returns:
            np.ndarray of shape (192,)
        """
        with self._lock:
            # Convert PCM_16 bytes → float32 tensor (model expects tensor, not file path)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_np = audio_np / 32768.0  # normalize to [-1, 1]
            audio_tensor = torch.from_numpy(audio_np).to(self.device)

            try:
                with torch.no_grad():
                    embedding = self.model(audio_tensor)

                if isinstance(embedding, dict):
                    embedding = embedding.get("spk_embedding", embedding.get("output", embedding))

                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy()

                embedding = embedding.flatten()

                if embedding.shape[0] != 192:
                    raise ValueError(
                        f"Expected 192-dim embedding, got {embedding.shape[0]}-dim."
                    )

                return embedding
            except Exception:
                raise

    def _match_voice(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Match embedding against all known voice embeddings using cosine similarity.
        Mirrors: face_recognition.py lines 230-254

        Returns:
            (face_id, score) if match >= threshold, else (None, best_score)
        """
        if self.known_embeddings.size == 0:
            _log_event("--", "VOICE MATCH", "DB empty, no match possible", _DIM)
            return None, 0.0

        embedding_2d = embedding.reshape(1, -1)
        sim = cosine_similarity(embedding_2d, self.known_embeddings)
        best_match_idx = np.argmax(sim)
        best_score = float(sim[0, best_match_idx])

        if best_score >= self.recognition_threshold:
            matched_id = self.known_ids[best_match_idx]
            _log_event("~~", "VOICE MATCH", f"{_BOLD}{matched_id}{_RESET}  score={best_score:.4f}", _GREEN)
            return matched_id, best_score
        else:
            best_name = self.known_ids[best_match_idx] if self.known_ids else "N/A"
            _log_event("xx", "NO MATCH", f"best={best_name}  score={best_score:.4f}  threshold={self.recognition_threshold}", _YELLOW)
            return None, best_score

    def _save_voice(self, face_id: str, embedding: np.ndarray,
                    audio_data: bytes = None, sample_rate: int = 16000):
        """
        Save a voice embedding to disk under the given face_id.
        Mirrors: face_recognition.py lines 256-281

        Saves:
            voice_db/face_N.npy  — the 192-dim embedding
            voice_db/face_N.wav  — the source audio (optional)
        """
        # Save embedding
        npy_path = self.db_dir / f"{face_id}.npy"
        np.save(npy_path, embedding)

        # Update in-memory arrays
        self.known_ids.append(face_id)
        if self.known_embeddings.size == 0:
            self.known_embeddings = embedding.reshape(1, -1)
        else:
            self.known_embeddings = np.vstack([self.known_embeddings, embedding.reshape(1, -1)])

        # Optionally save audio as WAV
        if audio_data is not None:
            wav_path = self.db_dir / f"{face_id}.wav"
            try:
                with wave.open(str(wav_path), 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data)
                _log_event(">>", "VOICE SAVED", f"{_BOLD}{face_id}{_RESET}  .npy + .wav", _MAGENTA)
            except Exception as e:
                logger.warning(f"Failed to save WAV for {face_id}: {e}")
                _log_event(">>", "VOICE SAVED", f"{_BOLD}{face_id}{_RESET}  .npy only", _MAGENTA)
        else:
            _log_event(">>", "VOICE SAVED", f"{_BOLD}{face_id}{_RESET}  .npy only", _MAGENTA)

    def has_voice(self, face_id: str) -> bool:
        """Check if a voice embedding exists for this face_id."""
        return face_id in self.known_ids

    def identify_speaker(self, audio_data: bytes, sample_rate: int = 16000,
                         current_face_id: str = None) -> dict:
        """
        Full pipeline: extract embedding -> match voice DB -> enroll if needed.

        Args:
            audio_data: Raw PCM_16 bytes (>= 0.5s)
            sample_rate: 16000
            current_face_id: The face_id of the person currently visible on camera.
                             Used to link voice to face when enrolling.

        Returns:
            {
                "face_id": str or None,     — matched/enrolled face_id
                "confidence": float,        — cosine similarity (0.0 if new enrollment)
                "is_new": bool,             — True if voice was just enrolled
                "status": str,              — human-readable status for logging
                "embedding": np.ndarray     — the 192-dim embedding
            }
        """
        # Minimum audio length check: 0.5s
        min_bytes = int(0.5 * sample_rate * 2)
        if len(audio_data) < min_bytes:
            duration = len(audio_data) / (sample_rate * 2)
            _log_event("!!", "TOO SHORT", f"{duration:.2f}s (min 0.5s)", _YELLOW)
            raise ValueError(
                f"Audio segment too short ({len(audio_data)} bytes, "
                f"minimum {min_bytes} bytes / 0.5s required)"
            )

        audio_duration = len(audio_data) / (sample_rate * 2)
        _log_event("<<", "IDENTIFY", f"audio={audio_duration:.2f}s  face={current_face_id or 'none'}", _CYAN)

        # Step 1: Extract embedding
        embedding = self.extract_embedding(audio_data, sample_rate)

        # Step 2: Match against voice DB
        matched_id, confidence = self._match_voice(embedding)

        if matched_id is not None:
            _log_event("++", "RECOGNIZED", f"{_BOLD}{matched_id}{_RESET}  confidence={confidence:.4f}", _GREEN)
            return {
                "face_id": matched_id,
                "confidence": confidence,
                "is_new": False,
                "status": f"recognized:{matched_id}",
                "embedding": embedding
            }

        # Step 3: No voice match — try to enroll using face_id
        if current_face_id is not None:
            if not self.has_voice(current_face_id):
                self._save_voice(current_face_id, embedding, audio_data, sample_rate)
                _log_event("**", "ENROLLED", f"{_BOLD}{current_face_id}{_RESET}  (face known, voice new)", _MAGENTA)
                return {
                    "face_id": current_face_id,
                    "confidence": 0.0,
                    "is_new": True,
                    "status": f"enrolled:{current_face_id}",
                    "embedding": embedding
                }
            else:
                _log_event("??", "UNKNOWN", f"voice mismatch for {current_face_id}  score={confidence:.4f}  (someone else speaking?)", _YELLOW)
                return {
                    "face_id": None,
                    "confidence": confidence,
                    "is_new": False,
                    "status": f"unknown:voice_mismatch_for_{current_face_id}",
                    "embedding": embedding
                }

        _log_event("??", "UNKNOWN", "no voice match, no face_id provided", _RED)
        return {
            "face_id": None,
            "confidence": confidence,
            "is_new": False,
            "status": "unknown:no_face_no_voice_match",
            "embedding": embedding
        }
