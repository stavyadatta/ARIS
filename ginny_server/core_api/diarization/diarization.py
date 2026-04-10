import os
import sys
import wave
import tempfile
import threading
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, List

# Add vendored pyannote-audio (DiariZen's fork with config/seg_duration/device params)
# and DiariZen itself to path. pip's pyannote.audio must be uninstalled.
sys.path.insert(0, "/workspace/diarization/DiariZen/pyannote-audio")
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
        device: torch.device = torch.device("cuda:2"),
        max_speakers: Optional[int] = None,
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

        # Runtime override of the global clustering speaker cap.
        # Without this, the value baked into config.toml (typically 4) is used.
        if max_speakers is not None:
            print(f'Overriding clustering max_speakers: '
                  f'{clustering_config.get("max_speakers")} -> {max_speakers}')
            clustering_config["max_speakers"] = max_speakers

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

    def call_with_posteriors(self, in_wav, sess_name=None):
        """
        Like pipeline.__call__() but also returns the raw soft powerset posteriors.
        Single forward pass of segmentation.

        The hard path is byte-identical to upstream inference.py:120-190 because:
          1. get_segmentations(soft=True) returns raw powerset probabilities.
          2. We replay self._segmentation.conversion(soft_data, soft=False) which
             IS the exact Powerset.to_multilabel call upstream's soft=False path
             runs internally (see pyannote-audio inference.py:225-230 and
             Powerset.forward at powerset.py:130).
          3. The resulting multilabel tensor (chunks, frames, num_local_speakers)
             is the SAME tensor upstream passes to median_filter and downstream.

        Returns:
            (result, soft_data_raw, sw)
            - result: pyannote Annotation, byte-identical to __call__() output.
            - soft_data_raw: np.ndarray (num_chunks, frames_per_chunk, num_powerset_classes),
              dtype float32, UNFILTERED. Consumers apply their own filter to a copy.
            - sw: SlidingWindow of the segmentation output (chunk stride).

        Body mirrors DiariZen inference.py:120-190; update in lockstep if upstream changes.
        Last verified against DiariZen commit: 510d2fe39cbf02e38b9e87ef2154e274d5ef9af0
        """
        from pyannote.audio.utils.signal import Binarize
        from scipy.ndimage import median_filter
        import torchaudio
        from pyannote.core import SlidingWindowFeature
        from pyannote.database import ProtocolFile

        assert self._segmentation.model.specifications.powerset is True, \
            "Model must be powerset-trained"
        assert hasattr(self, 'apply_median_filtering'), \
            "Pipeline must have apply_median_filtering attribute (DiariZen contract)"
        assert hasattr(self._segmentation, 'conversion'), \
            "Inference wrapper must have conversion attribute (pyannote Inference contract)"

        in_wav = in_wav if not isinstance(in_wav, ProtocolFile) else in_wav['audio']
        waveform, sample_rate = torchaudio.load(in_wav)
        waveform = torch.unsqueeze(waveform[0], 0)

        # *** SINGLE SOFT FORWARD PASS ***
        soft_segmentations = self.get_segmentations(
            {"waveform": waveform, "sample_rate": sample_rate},
            soft=True
        )
        soft_data_raw = np.asarray(soft_segmentations.data, dtype=np.float32)
        sw = soft_segmentations.sliding_window

        # *** BYTE-IDENTICAL HARD PATH: replay upstream's Powerset.to_multilabel ***
        # self._segmentation.conversion is the Powerset instance upstream already
        # uses when you call get_segmentations(soft=False). Calling it with
        # soft=False does: argmax -> one_hot -> matmul(mapping) -> multilabel.
        # Result shape: (chunks, frames, num_local_speakers). Binary float.
        #
        # Byte-identity proof: upstream's soft=True path returns exp(logits) per
        # Powerset.to_multilabel. argmax is monotone under exp, so
        # argmax(exp(x)) == argmax(x). Therefore calling
        # conversion(exp(logits), soft=False) yields the same argmax and thus
        # the same multilabel projection as conversion(logits, soft=False).
        conversion = self._segmentation.conversion
        # Device alignment for the conversion call:
        # - The Inference wrapper stores the canonical device on `self._segmentation.device`
        #   and uses it to move the model in __init__ (`self.model.to(self.device)`).
        # - But it does NOT move `self.conversion` to the device in __init__ — that
        #   only happens in a separate `to()` method that may never be invoked.
        # - So `conversion.mapping` may be on CPU even when the model is on cuda:2.
        # - Note: pyannote's Model class does NOT expose a `.device` attribute, so
        #   we use the Inference wrapper's `.device` as the source of truth, with a
        #   safe fallback to the first model parameter's device.
        try:
            target_device = self._segmentation.device
        except AttributeError:
            target_device = next(self._segmentation.model.parameters()).device
        if conversion.mapping.device != target_device:
            conversion.to(target_device)
        soft_tensor = torch.from_numpy(soft_data_raw).to(target_device)
        multilabel_hard = conversion(soft_tensor, soft=False).cpu().numpy()
        binarized_segmentations = SlidingWindowFeature(multilabel_hard, sw)

        # *** MEDIAN FILTER: exact upstream parity (inference.py:129-130) ***
        if self.apply_median_filtering:
            binarized_segmentations.data = median_filter(
                binarized_segmentations.data, size=(1, 11, 1), mode='reflect'
            )

        # Replay upstream inference.py:135-183 verbatim on the multilabel tensor
        count = self.speaker_count(
            binarized_segmentations,
            self._segmentation.model._receptive_field,
            warm_up=(0.0, 0.0),
        )

        embeddings = self.get_embeddings(
            {"waveform": waveform, "sample_rate": sample_rate},
            binarized_segmentations,
            exclude_overlap=self.embedding_exclude_overlap,
        )

        hard_clusters, _, _ = self.clustering(
            embeddings=embeddings,
            segmentations=binarized_segmentations,
            min_clusters=self.min_speakers,
            max_clusters=self.max_speakers
        )

        count.data = np.minimum(count.data, self.max_speakers).astype(np.int8)
        inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
        hard_clusters[inactive_speakers] = -2
        discrete_diarization, _ = self.reconstruct(
            binarized_segmentations,
            hard_clusters,
            count,
        )

        to_annotation = Binarize(
            onset=0.5, offset=0.5, min_duration_on=0.0, min_duration_off=0.0
        )
        result = to_annotation(discrete_diarization)
        result.uri = sess_name

        if self.rttm_out_dir is not None and sess_name is not None:
            rttm_out = os.path.join(self.rttm_out_dir, sess_name + ".rttm")
            with open(rttm_out, "w") as f:
                f.write(result.to_rttm())

        return result, soft_data_raw, sw

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        cache_dir: str = None,
        rttm_out_dir: str = None,
        device: torch.device = torch.device("cuda:2"),
        max_speakers: Optional[int] = None,
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
            max_speakers=max_speakers,
        )


class _Diarization:
    """
    Diarization module wrapping DiariZen with device override.

    Provides two modes:
    1. diarize(audio_data) - Full diarization on a single audio buffer
    2. Session-based buffer management for accumulating short segments

    Thread-safe via Lock.
    """

    MIN_DIARIZATION_DURATION = 20.0

    def __init__(self,
                 model_name: str = "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
                 device: str = "cuda:2",
                 max_speakers: int = 8):
        self.device = torch.device(device)
        self.max_speakers = max_speakers
        self._lock = threading.Lock()

        # Load DiariZen with device override and runtime max_speakers cap.
        self.pipeline = _DiariZenCUDA1.from_pretrained(
            model_name,
            device=self.device,
            max_speakers=max_speakers,
        )

        # *** Powerset discovery (Phase 2) ***
        # Verified attribute paths against vendored source:
        #   inference.py:154 - self.conversion = conversion[0] (Powerset instance on Inference wrapper)
        #   task.py:103 - powerset_max_classes is a flat int attribute on Specifications
        #   powerset.py:48-53 - mapping and cardinality registered as buffers
        seg_inference = self.pipeline._segmentation
        seg_model = seg_inference.model
        specs = seg_model.specifications

        if not getattr(specs, 'powerset', False):
            raise RuntimeError(
                "Loaded segmentation model is not a powerset model. "
                "This pipeline requires a powerset-trained DiariZen checkpoint."
            )

        num_local_classes = len(specs.classes)
        max_per_frame = specs.powerset_max_classes
        num_powerset_classes = specs.num_powerset_classes

        if not hasattr(seg_inference, 'conversion'):
            raise RuntimeError(
                "Inference wrapper has no 'conversion' attribute. "
                "This pyannote version is incompatible with the per-frame gate."
            )
        conversion = seg_inference.conversion
        if not hasattr(conversion, 'mapping') or not hasattr(conversion, 'cardinality'):
            raise RuntimeError(
                f"Conversion object {type(conversion).__name__} has no mapping/cardinality. "
                f"Expected a Powerset instance; got something else."
            )

        mapping = conversion.mapping.detach().cpu().numpy().astype(np.int8)
        cardinality = conversion.cardinality.detach().cpu().numpy().astype(np.int64)

        if mapping.shape != (num_powerset_classes, num_local_classes):
            raise RuntimeError(
                f"Powerset mapping shape mismatch: expected "
                f"({num_powerset_classes}, {num_local_classes}), got {mapping.shape}"
            )

        single_speaker_class_idxs = np.where(cardinality == 1)[0].tolist()
        # Filter out any indices that are out of bounds for num_powerset_classes
        single_speaker_class_idxs = [idx for idx in single_speaker_class_idxs if idx < num_powerset_classes]
        silence_idxs = np.where(cardinality == 0)[0].tolist()
        overlap_class_idxs = np.where(cardinality >= 2)[0].tolist()

        if len(single_speaker_class_idxs) != num_local_classes:
            raise RuntimeError(
                f"Powerset expected {num_local_classes} single-speaker classes, "
                f"got {len(single_speaker_class_idxs)}"
            )
        if len(silence_idxs) != 1:
            raise RuntimeError(
                f"Powerset expected 1 silence class, got {len(silence_idxs)}"
            )

        self.powerset_class_map = {
            "single_speaker_class_idxs": single_speaker_class_idxs,
            "silence_class_idx": silence_idxs[0],
            "overlap_class_idxs": overlap_class_idxs,
        }

        # Frame rate via public attribute first, private fallback
        try:
            frame_step = seg_model.example_output.frames.step
            frame_rate_source = "example_output.frames"
        except AttributeError:
            frame_step = seg_model._receptive_field.step
            frame_rate_source = "_receptive_field"
        self.segmentation_frame_rate_hz = 1.0 / frame_step

        print(f"[diarization] model classes: num_local={num_local_classes}, "
              f"max_per_frame={max_per_frame}, num_powerset_classes={num_powerset_classes}")
        print(f"[diarization] powerset cardinality: single={single_speaker_class_idxs}, "
              f"silence={silence_idxs[0]}, overlap={overlap_class_idxs}")
        print(f"[diarization] segmentation frame rate: {self.segmentation_frame_rate_hz:.2f}Hz "
              f"(source: {frame_rate_source})")

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

    def diarize_with_posteriors(self, audio_data: bytes, sample_rate: int = 16000):
        """
        Like diarize() but also returns soft powerset posteriors and metadata
        needed by the per-frame enrollment gate.

        Returns:
            (segments, soft_data_raw, sw, powerset_class_map, frame_rate_hz)
            - segments: identical to diarize() return
            - soft_data_raw: np.ndarray (num_chunks, frames_per_chunk, num_powerset_classes), float32
            - sw: SlidingWindow of segmentation chunks (window-local times)
            - powerset_class_map: cached dict from __init__
            - frame_rate_hz: cached float from __init__
        """
        duration = self.get_audio_duration(audio_data, sample_rate)
        if duration < self.MIN_DIARIZATION_DURATION:
            raise ValueError(
                f"Audio too short for diarization ({duration:.1f}s, "
                f"minimum {self.MIN_DIARIZATION_DURATION}s)."
            )

        with self._lock:
            temp_wav = self._save_to_temp_wav(audio_data, sample_rate)
            try:
                result, soft_data_raw, sw = self.pipeline.call_with_posteriors(temp_wav)
            finally:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)

        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        segments = []
        for turn, _, speaker in result.itertracks(yield_label=True):
            start_sample = max(0, int(turn.start * sample_rate))
            end_sample = min(len(audio_np), int(turn.end * sample_rate))
            if end_sample <= start_sample:
                continue
            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
                "audio": audio_np[start_sample:end_sample].tobytes()
            })

        return (
            segments, soft_data_raw, sw,
            self.powerset_class_map, self.segmentation_frame_rate_hz,
        )

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
