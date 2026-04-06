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

# Set up logger for speaker recognition (prevent duplicate handlers)
logger = logging.getLogger("speaker_recognition")
logger.setLevel(logging.INFO)
logger.propagate = False  # Don't bubble up to root logger (prevents double printing)

if not logger.handlers:
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


# ===================== Model Registry =====================

_DB_ROOT = "/workspace/database/embedding_accumulation_method"

MODEL_REGISTRY = {
    "eres2netv2": {
        "display_name": "ERes2NetV2",
        "model_id": "iic/speech_eres2netv2_sv_zh-cn_16k-common",
        "db_dir": f"{_DB_ROOT}/voice_eres2netv2",
        "emb_dim": 192,
    },
    "titanet": {
        "display_name": "TitaNet Large",
        "model_id": "nvidia/speakerverification_en_titanet_large",
        "db_dir": f"{_DB_ROOT}/voice_titanet",
        "emb_dim": 192,
    },
    "redimnet": {
        "display_name": "ReDimNet B6",
        "model_id": "B6",
        "db_dir": f"{_DB_ROOT}/voice_redimnet",
        "emb_dim": 192,
    },
    "wavlm_ssl": {
        "display_name": "WavLM-MHFA (SSL-SV)",
        "model_id": "wavlm_mhfa",
        "db_dir": f"{_DB_ROOT}/voice_wavlm_ssl",
        "emb_dim": 256,
    },
}


class _SpeakerRecognition:
    """
    Speaker recognition with switchable models (16kHz input).

    Supported models (selected via model_name):
    - eres2netv2: ERes2NetV2 from ModelScope (192-dim)
    - titanet: NVIDIA TitaNet Large from NeMo (192-dim)
    - redimnet: ReDimNet B6 from IDRnD via torch.hub (192-dim)
    - wavlm_ssl: WavLM-Base+ MHFA from theolepage/wavlm_ssl_sv (256-dim)

    Each model has its own embedding directory under /workspace/database/.
    Matches via in-memory cosine similarity (sklearn).
    Thread-safe via Lock.
    """

    def __init__(self,
                 model_name: str = "eres2netv2",
                 recognition_threshold: float = 0.7,
                 device: str = "cuda:1"):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {', '.join(MODEL_REGISTRY.keys())}"
            )

        self.model_name = model_name
        self.device = device
        config = MODEL_REGISTRY[model_name]
        self.db_dir = Path(config["db_dir"])
        self.recognition_threshold = recognition_threshold
        self.emb_dim = config["emb_dim"]
        self._lock = threading.Lock()
        self._display_name = config["display_name"]

        # Ensure embedding directory exists
        self._ensure_db_directory()

        # Load the selected model
        _log_event("...", "MODEL LOAD", f"{self._display_name} on {device}", _DIM)
        loader = getattr(self, f"_load_{model_name}")
        loader(config["model_id"], device)
        _log_event("OK", "MODEL READY", f"{self._display_name} on {device}", _GREEN)

        # Load existing voice embeddings from disk
        self.known_ids, self.known_embeddings = self._load_database()
        _log_event("DB", "VOICE DB",
                   f"{len(self.known_ids)} voices loaded from {self.db_dir}  "
                   f"[{self._display_name}]", _CYAN)

    # ===================== Model Loaders =====================

    def _load_eres2netv2(self, model_id: str, device: str):
        """Load ERes2NetV2 via ModelScope."""
        from modelscope.models import Model
        self.model = Model.from_pretrained(model_id, device=device)
        self.model.eval()

    def _load_titanet(self, model_id: str, device: str):
        """Load TitaNet Large via NVIDIA NeMo."""
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            raise ImportError(
                "NeMo toolkit is required for TitaNet. "
                "Install with: pip install nemo_toolkit[asr]"
            )
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_id)
        self.model = self.model.to(device)
        self.model.eval()

    def _load_redimnet(self, model_id: str, device: str):
        """Load ReDimNet B6 via torch.hub."""
        self.model = torch.hub.load(
            "IDRnD/ReDimNet", "ReDimNet",
            model_name=model_id,
            train_type="ft_lm",
            dataset="vox2",
        )
        self.model = self.model.to(device)
        self.model.eval()

    def _load_wavlm_ssl(self, model_id: str, device: str):
        """Load WavLM-Base+ MHFA speaker encoder from wavlm_ssl_sv."""
        import sys as _sys
        _model_dir = Path(__file__).parent.parent.parent / "models" / "wavlm_ssl_sv"
        _sys.path.insert(0, str(_model_dir))

        from Baseline.Spk_Encoder import MainModel

        wavlm_pt = _model_dir / "WavLM-Base+.pt"
        checkpoint_path = _model_dir / "model000000018.model"

        if not wavlm_pt.exists():
            raise FileNotFoundError(
                f"WavLM-Base+ not found at {wavlm_pt}. "
                f"Download from: https://github.com/microsoft/unilm/tree/master/wavlm"
            )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Speaker model checkpoint not found at {checkpoint_path}. "
                f"Download from: https://drive.google.com/drive/folders/"
                f"1ygZPvdGwepWDDfIQp6aPRktt2QxLt6cE"
            )

        # Build model (loads WavLM-Base+ internally)
        self.model = MainModel(
            pretrained_model_path=str(wavlm_pt),
            nOut=256,
            weight_finetuning_reg=0.01,
        )

        # Load fine-tuned speaker verification weights
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
        # Handle wrapped model state dicts
        cleaned = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "").replace("__S__.", "")
            cleaned[k] = v
        self.model.load_state_dict(cleaned, strict=False)

        self.model = self.model.to(device)
        self.model.eval()

    # ===================== Embedding Extractors =====================

    def _extract_eres2netv2(self, audio_np: np.ndarray) -> np.ndarray:
        """Extract embedding using ERes2NetV2 (ModelScope)."""
        audio_tensor = torch.from_numpy(audio_np).to(self.device)
        with torch.no_grad():
            embedding = self.model(audio_tensor)

        if isinstance(embedding, dict):
            embedding = embedding.get("spk_embedding", embedding.get("output", embedding))
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        return embedding.flatten()

    def _extract_titanet(self, audio_np: np.ndarray) -> np.ndarray:
        """Extract embedding using TitaNet Large (NeMo)."""
        try:
            # Preferred: direct numpy input (NeMo >= 1.23)
            emb = self.model.infer_segment(audio_np)
        except (AttributeError, TypeError):
            # Fallback: write temp WAV for older NeMo versions
            wav_path = self._float32_to_temp_wav(audio_np)
            try:
                emb = self.model.get_embedding(wav_path)
            finally:
                os.remove(wav_path)
        # NeMo may return (embedding, logits) tuple
        if isinstance(emb, tuple):
            emb = emb[0]
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        return emb.flatten()

    def _extract_redimnet(self, audio_np: np.ndarray) -> np.ndarray:
        """Extract embedding using ReDimNet B6 (torch.hub)."""
        waveform = torch.from_numpy(audio_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(waveform)
        return emb.squeeze(0).cpu().float().numpy().flatten()

    def _extract_wavlm_ssl(self, audio_np: np.ndarray) -> np.ndarray:
        """Extract 256-dim embedding using WavLM-Base+ MHFA."""
        waveform = torch.from_numpy(audio_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model([waveform, "test"])
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        return emb.flatten()

    # ===================== Public API =====================

    def _ensure_db_directory(self):
        """Ensure that the voice database directory exists."""
        if not self.db_dir.exists():
            self.db_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created voice DB directory: {self.db_dir}")

    def _load_database(self) -> Tuple[List[str], np.ndarray]:
        """
        Load known voice embeddings and IDs from the database directory.

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

    def _float32_to_temp_wav(self, audio_np: np.ndarray, sample_rate: int = 16000) -> str:
        """Save float32 numpy array to a temporary WAV file. Returns path."""
        import soundfile as sf
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file_path = temp_file.name
        temp_file.close()
        try:
            sf.write(temp_file_path, audio_np.astype(np.float32), sample_rate)
            return temp_file_path
        except Exception:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise

    def extract_embedding(self, audio_data: bytes, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract 192-dim embedding from raw PCM_16 audio bytes.
        Uses the active model selected at init time.
        Thread-safe via Lock.

        Returns:
            np.ndarray of shape (192,)
        """
        with self._lock:
            # Convert PCM_16 bytes → float32 numpy, normalized to [-1, 1]
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_np = audio_np / 32768.0

            extractor = getattr(self, f"_extract_{self.model_name}")
            embedding = extractor(audio_np)

            if embedding.shape[0] != self.emb_dim:
                raise ValueError(
                    f"Expected {self.emb_dim}-dim embedding, got {embedding.shape[0]}-dim "
                    f"from {self._display_name}."
                )

            return embedding

    def _match_voice(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Match embedding against all known voice embeddings using cosine similarity.

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

        Saves:
            {db_dir}/face_N.npy  — the 192-dim embedding
            {db_dir}/face_N.wav  — the source audio (optional)
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

    def _generate_voice_id(self) -> str:
        """Generate a new voice_N ID (sequential, like face_N)."""
        current_ids = [int(x.replace("voice_", ""))
                       for x in self.known_ids if x.startswith("voice_")]
        next_id = (max(current_ids) + 1) if current_ids else 1
        return f"voice_{next_id}"

    # ===================== Multi-Sample Enrollment Buffer =====================

    MIN_ENROLL_DURATION = 20.0   # seconds — minimum total speech before enrollment
    MAX_PENDING_AUDIO = 60.0     # seconds — hard cap on accumulated audio per pending entry

    # Adaptive pending-buffer match threshold:
    #   threshold = min(BASE + SLOPE * total_duration, MAX)
    # Permissive when noisy (early samples), strict (matches DB threshold) once stable.
    PENDING_THRESHOLD_BASE = 0.5
    PENDING_THRESHOLD_SLOPE = 0.01
    PENDING_THRESHOLD_MAX = 0.7

    @classmethod
    def _adaptive_pending_threshold(cls, total_duration: float) -> float:
        """Linear ramp from BASE to MAX over total_duration seconds of speech."""
        return min(
            cls.PENDING_THRESHOLD_BASE + cls.PENDING_THRESHOLD_SLOPE * total_duration,
            cls.PENDING_THRESHOLD_MAX,
        )

    def init_enrollment_buffer(self):
        """Initialize/reset the pending enrollment buffer.

        Schema per pending entry:
            {
                "embeddings":         [np.ndarray, ...],  # for running-average match
                "accumulated_audio":  bytearray,           # raw concat of cluster audio
                "total_duration":     float,               # speech seconds (post-diar)
                "capped":             bool,                # True once MAX_PENDING_AUDIO reached
            }
        """
        self._pending = {}
        self._next_pending_id = 1
        _log_event("--", "ENROLL BUF", "Enrollment buffer initialized", _DIM)

    def match_or_buffer(self, embedding: np.ndarray, audio_data: bytes = None,
                        sample_rate: int = 16000, min_samples: int = 3) -> dict:
        """
        Match embedding against voice DB. If no match, add to pending enrollment buffer.

        Pending entries accumulate cluster audio (raw concat, no padding) until they
        pass the gate (>=min_samples AND >=MIN_ENROLL_DURATION speech). On gate pass,
        the accumulated audio is re-extracted through the speaker encoder once to
        produce the final stitched embedding.

        Pending-buffer match uses an adaptive threshold that scales with the
        candidate entry's total_duration (see _adaptive_pending_threshold).

        Returns:
            {
                "voice_id": str or None,
                "confidence": float,
                "is_new": bool,
                "is_pending": bool,
                "pending_id": str or None,
                "status": str
            }
        """
        if not hasattr(self, '_pending'):
            self.init_enrollment_buffer()

        # Step 1: Match against voice DB
        matched_id, confidence = self._match_voice(embedding)
        if matched_id is not None:
            return {
                "voice_id": matched_id,
                "confidence": confidence,
                "is_new": False,
                "is_pending": False,
                "pending_id": None,
                "status": f"recognized:{matched_id}"
            }

        # Step 2: Match against pending buffer entries (running average per entry)
        best_pending_id = None
        best_pending_score = 0.0
        emb_2d = embedding.reshape(1, -1)

        for pid, entry in self._pending.items():
            avg_emb = np.mean(entry["embeddings"], axis=0).reshape(1, -1)
            score = float(cosine_similarity(emb_2d, avg_emb)[0, 0])
            if score > best_pending_score:
                best_pending_score = score
                best_pending_id = pid

        if best_pending_id is not None:
            # Adaptive threshold computed against the CANDIDATE entry's total_duration
            candidate_dur = self._pending[best_pending_id]["total_duration"]
            adaptive_thresh = self._adaptive_pending_threshold(candidate_dur)
        else:
            adaptive_thresh = self.PENDING_THRESHOLD_BASE

        if best_pending_id and best_pending_score >= adaptive_thresh:
            entry = self._pending[best_pending_id]

            if entry["capped"]:
                # Cap reached — drop new audio/embedding silently (single warn per cap hit)
                _log_event("..", "CAP HIT",
                           f"{_BOLD}{best_pending_id}{_RESET}  "
                           f"audio frozen at {entry['total_duration']:.1f}s, "
                           f"new cluster ignored", _DIM)
                return {
                    "voice_id": None,
                    "confidence": best_pending_score,
                    "is_new": False,
                    "is_pending": True,
                    "pending_id": best_pending_id,
                    "status": f"pending_capped:{best_pending_id}"
                }

            # Append embedding for running-average match
            entry["embeddings"].append(embedding)

            # Append raw audio bytes (no silence, no padding)
            if audio_data:
                dur = len(audio_data) / (sample_rate * 2)
                entry["accumulated_audio"].extend(audio_data)
                entry["total_duration"] += dur

                if entry["total_duration"] >= self.MAX_PENDING_AUDIO:
                    entry["capped"] = True
                    _log_event("!!", "CAP REACHED",
                               f"{_BOLD}{best_pending_id}{_RESET}  "
                               f"{entry['total_duration']:.1f}s "
                               f">= {self.MAX_PENDING_AUDIO:.0f}s, freezing", _YELLOW)

            n_samples = len(entry['embeddings'])
            total_dur = entry["total_duration"]
            _log_event("..", "PENDING",
                       f"{_BOLD}{best_pending_id}{_RESET}  "
                       f"{n_samples}/{min_samples} samples  "
                       f"{total_dur:.1f}/{self.MIN_ENROLL_DURATION:.0f}s  "
                       f"match={best_pending_score:.2f} thr={adaptive_thresh:.2f}",
                       _CYAN)

            # Gate: enough samples AND enough speech
            if n_samples >= min_samples and total_dur >= self.MIN_ENROLL_DURATION:
                return self._flush_enrollment(best_pending_id, sample_rate)

            return {
                "voice_id": None,
                "confidence": best_pending_score,
                "is_new": False,
                "is_pending": True,
                "pending_id": best_pending_id,
                "status": f"pending:{best_pending_id}:{n_samples}/{min_samples}:{total_dur:.0f}/{self.MIN_ENROLL_DURATION:.0f}s"
            }

        # Step 3: No match in DB or buffer — create new pending entry
        pid = f"pending_{self._next_pending_id}"
        self._next_pending_id += 1
        dur = len(audio_data) / (sample_rate * 2) if audio_data else 0.0
        accum = bytearray()
        if audio_data:
            accum.extend(audio_data)
        self._pending[pid] = {
            "embeddings": [embedding],
            "accumulated_audio": accum,
            "total_duration": dur,
            "capped": dur >= self.MAX_PENDING_AUDIO,
        }

        _log_event("++", "NEW PENDING",
                   f"{_BOLD}{pid}{_RESET}  1/{min_samples} samples  "
                   f"{dur:.1f}s  (new voice detected)", _MAGENTA)

        return {
            "voice_id": None,
            "confidence": 0.0,
            "is_new": False,
            "is_pending": True,
            "pending_id": pid,
            "status": f"pending:{pid}:1/{min_samples}"
        }

    def _flush_enrollment(self, pending_id: str, sample_rate: int = 16000) -> dict:
        """Enroll a pending entry by re-extracting an embedding from the FULL
        accumulated (stitched) audio, then saving both the embedding and the WAV.

        This replaces the old "average of per-cluster embeddings" approach: the
        speaker encoder gets a longer, richer audio context to produce a single
        prototype embedding, which is more reliable than averaging short-clip
        embeddings.
        """
        entry = self._pending.pop(pending_id)
        accumulated_audio = bytes(entry["accumulated_audio"])
        n_samples = len(entry["embeddings"])
        total_dur = entry["total_duration"]

        # Re-extract embedding from the full stitched audio
        stitched_embedding = self.extract_embedding(accumulated_audio, sample_rate)

        # L2 normalize
        norm = np.linalg.norm(stitched_embedding)
        if norm > 0:
            stitched_embedding = stitched_embedding / norm

        # Generate voice_id and save (saves npy + the FULL stitched WAV)
        voice_id = self._generate_voice_id()
        self._save_voice(voice_id, stitched_embedding, accumulated_audio, sample_rate)

        _log_event("**", "ENROLLED",
                   f"{_BOLD}{voice_id}{_RESET}  from {pending_id}  "
                   f"({n_samples} samples, {total_dur:.1f}s stitched, "
                   f"{len(accumulated_audio)} bytes)", _GREEN)

        return {
            "voice_id": voice_id,
            "confidence": 0.0,
            "is_new": True,
            "is_pending": False,
            "pending_id": None,
            "status": f"enrolled:{voice_id}:from_{pending_id}"
        }

    def flush_all_pending(self, sample_rate: int = 16000, min_samples: int = 1):
        """Force-enroll all pending entries (e.g., at end of video). Entries with
        fewer than min_samples or less than MIN_ENROLL_DURATION total audio
        are discarded."""
        if not hasattr(self, '_pending'):
            return []

        results = []
        for pid in list(self._pending.keys()):
            entry = self._pending[pid]
            n = len(entry["embeddings"])
            total_dur = entry.get("total_duration", 0.0)
            if n >= min_samples and total_dur >= self.MIN_ENROLL_DURATION:
                result = self._flush_enrollment(pid, sample_rate)
                results.append(result)
            else:
                reason = []
                if n < min_samples:
                    reason.append(f"{n}/{min_samples} samples")
                if total_dur < self.MIN_ENROLL_DURATION:
                    reason.append(f"{total_dur:.1f}/{self.MIN_ENROLL_DURATION:.0f}s audio")
                _log_event("--", "DISCARD",
                           f"{pid}  {', '.join(reason)}", _DIM)
                del self._pending[pid]
        return results

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

        # No face_id provided, no voice match — auto-enroll with voice_N ID
        new_voice_id = self._generate_voice_id()
        self._save_voice(new_voice_id, embedding, audio_data, sample_rate)
        _log_event("**", "ENROLLED", f"{_BOLD}{new_voice_id}{_RESET}  (new voice, no face)", _MAGENTA)
        return {
            "face_id": new_voice_id,
            "confidence": 0.0,
            "is_new": True,
            "status": f"enrolled:{new_voice_id}",
            "embedding": embedding
        }
