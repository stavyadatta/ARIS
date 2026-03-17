import os
import sys
import wave
import tempfile
import threading
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, List

# Add DiariZen to path so we can import its modules
sys.path.insert(0, "/workspace/diarization/DiariZen")

from diarizen.pipelines.inference import DiariZenPipeline
from huggingface_hub import snapshot_download, hf_hub_download


class _DiariZenCUDA1(DiariZenPipeline):
    """
    Subclass of DiariZenPipeline that overrides the hardcoded cuda:0 device.

    DiariZenPipeline.__init__() hardcodes device=torch.device("cuda:0") at inference.py:56.
    The parent class SpeakerDiarization.__init__() accepts a device parameter and
    correctly passes it to Inference and PretrainedSpeakerEmbedding.

    This subclass intercepts __init__ to pass the correct device.
    """

    def __init__(
        self,
        diarizen_hub,
        embedding_model,
        config_parse: Optional[Dict] = None,
        rttm_out_dir: Optional[str] = None,
        device: torch.device = torch.device("cuda:1"),
    ):
        import toml
        from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline

        config_path = Path(diarizen_hub / "config.toml")
        config = toml.load(config_path.as_posix())

        if config_parse is not None:
            print('Overriding with parsed config.')
            config["inference"]["args"] = config_parse["inference"]["args"]
            config["clustering"]["args"] = config_parse["clustering"]["args"]

        inference_config = config["inference"]["args"]
        clustering_config = config["clustering"]["args"]

        print(f'Loaded configuration: {config}')

        # Call the GRANDPARENT (SpeakerDiarization) __init__ with our device
        # This bypasses DiariZenPipeline.__init__ which hardcodes cuda:0
        SpeakerDiarizationPipeline.__init__(
            self,
            config=config,
            seg_duration=inference_config["seg_duration"],
            segmentation=str(Path(diarizen_hub / "pytorch_model.bin")),
            segmentation_step=inference_config["segmentation_step"],
            embedding=embedding_model,
            embedding_exclude_overlap=True,
            clustering=clustering_config["method"],
            embedding_batch_size=inference_config["batch_size"],
            segmentation_batch_size=inference_config["batch_size"],
            device=device  # THE FIX: pass our device instead of hardcoded cuda:0
        )

        self.apply_median_filtering = inference_config["apply_median_filtering"]
        self.min_speakers = clustering_config["min_speakers"]
        self.max_speakers = clustering_config["max_speakers"]

        if clustering_config["method"] == "AgglomerativeClustering":
            self.PIPELINE_PARAMS = {
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": clustering_config["min_cluster_size"],
                    "threshold": clustering_config["ahc_threshold"],
                }
            }
        elif clustering_config["method"] == "VBxClustering":
            self.PIPELINE_PARAMS = {
                "clustering": {
                    "ahc_criterion": clustering_config["ahc_criterion"],
                    "ahc_threshold": clustering_config["ahc_threshold"],
                    "Fa": clustering_config["Fa"],
                    "Fb": clustering_config["Fb"],
                }
            }
            self.clustering.plda_dir = str(Path(diarizen_hub / "plda"))
            self.clustering.lda_dim = clustering_config["lda_dim"]
            self.clustering.maxIters = clustering_config["max_iters"]
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_config['method']}")

        self.instantiate(self.PIPELINE_PARAMS)

        if rttm_out_dir is not None:
            os.makedirs(rttm_out_dir, exist_ok=True)
        self.rttm_out_dir = rttm_out_dir

        assert self._segmentation.model.specifications.powerset is True

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        cache_dir: str = None,
        rttm_out_dir: str = None,
        device: torch.device = torch.device("cuda:1"),
    ) -> "_DiariZenCUDA1":
        diarizen_hub = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=cache_dir is not None
        )

        embedding_model = hf_hub_download(
            repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
            filename="pytorch_model.bin",
            cache_dir=cache_dir,
            local_files_only=cache_dir is not None
        )

        return cls(
            diarizen_hub=Path(diarizen_hub).expanduser().absolute(),
            embedding_model=embedding_model,
            rttm_out_dir=rttm_out_dir,
            device=device,
        )


class _Diarization:
    """
    Diarization module wrapping DiariZen with device override.

    Provides two modes:
    1. diarize(audio_data) - Full diarization on a single audio buffer
    2. Session-based buffer management for accumulating short segments

    Thread-safe via Lock.
    """

    MIN_DIARIZATION_DURATION = 10.0

    def __init__(self,
                 model_name: str = "BUT-FIT/diarizen-wavlm-large-s80-md",
                 device: str = "cuda:1",
                 max_speakers: int = 3):
        self.device = torch.device(device)
        self.max_speakers = max_speakers
        self._lock = threading.Lock()

        # Load DiariZen with device override
        self.pipeline = _DiariZenCUDA1.from_pretrained(
            model_name,
            device=self.device
        )

        # Per-session audio buffers
        self._session_buffers: Dict[str, dict] = {}
        self._buffer_lock = threading.Lock()

    def _save_to_temp_wav(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Save raw PCM_16 bytes to a temporary WAV file."""
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

    def get_audio_duration(self, audio_data: bytes, sample_rate: int = 16000) -> float:
        """Calculate duration in seconds from PCM_16 audio bytes."""
        num_samples = len(audio_data) // 2
        return num_samples / sample_rate

    def diarize(self, audio_data: bytes, sample_rate: int = 16000) -> List[dict]:
        """
        Run full diarization on audio buffer.
        Only call on audio >= MIN_DIARIZATION_DURATION (10s).

        Returns:
            List of {"speaker": str, "start": float, "end": float, "audio": bytes}
        """
        duration = self.get_audio_duration(audio_data, sample_rate)
        if duration < self.MIN_DIARIZATION_DURATION:
            raise ValueError(
                f"Audio too short for diarization ({duration:.1f}s, "
                f"minimum {self.MIN_DIARIZATION_DURATION}s). "
                f"Use single-speaker path for short segments."
            )

        with self._lock:
            temp_wav = self._save_to_temp_wav(audio_data, sample_rate)
            try:
                diar_result = self.pipeline(temp_wav)
            finally:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)

        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        segments = []
        for turn, _, speaker in diar_result.itertracks(yield_label=True):
            start_sample = int(turn.start * sample_rate)
            end_sample = int(turn.end * sample_rate)

            start_sample = max(0, start_sample)
            end_sample = min(len(audio_np), end_sample)

            if end_sample <= start_sample:
                continue

            segment_audio = audio_np[start_sample:end_sample].tobytes()
            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
                "audio": segment_audio
            })

        return segments

    def accumulate_segment(self, session_id: str, audio_data: bytes,
                           segment_start_time: float, sample_rate: int = 16000) -> Optional[List[dict]]:
        """
        Accumulate a VAD segment into the session buffer.
        When buffer exceeds MIN_DIARIZATION_DURATION, run diarization.

        Returns:
            None if buffer not yet long enough,
            List[dict] of diarized segments if buffer was processed
        """
        with self._buffer_lock:
            if session_id not in self._session_buffers:
                self._session_buffers[session_id] = {
                    "audio": bytearray(),
                    "start_time": segment_start_time,
                    "sample_rate": sample_rate
                }

            buf = self._session_buffers[session_id]
            buf["audio"].extend(audio_data)

            buffer_duration = self.get_audio_duration(bytes(buf["audio"]), sample_rate)

            if buffer_duration < self.MIN_DIARIZATION_DURATION:
                return None

            full_audio = bytes(buf["audio"])
            buffer_start = buf["start_time"]
            self._session_buffers[session_id] = {
                "audio": bytearray(),
                "start_time": segment_start_time,
                "sample_rate": sample_rate
            }

        # Run diarization outside buffer lock
        segments = self.diarize(full_audio, sample_rate)

        # Adjust timestamps to absolute time
        for seg in segments:
            seg["start"] += buffer_start
            seg["end"] += buffer_start

        return segments

    def clear_session(self, session_id: str):
        """Clean up session buffer when conversation ends."""
        with self._buffer_lock:
            self._session_buffers.pop(session_id, None)
