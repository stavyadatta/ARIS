"""BRAID perception orchestrator.

Runs face detection + face embedding + face quality (§2.1), Light-ASD
(§2.1 visual-audio synchrony), DiariZenCUDA1 diarization (§2.2), and WavLM
voice embeddings (§2.2) on a single 30s bundle.

Exposes ``PerceptionEngine.run(bundle) -> RawObservations``. Heavy models are
instantiated lazily; if weights or CUDA are unavailable on the executing
machine, the engine falls back to safe defaults so the rest of the pipeline
(posterior, decision) remains reachable during bring-up.
"""
from __future__ import annotations

import io
import logging
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import BraidConfig
from .log_style import C

logger = logging.getLogger("braid")


# ----- lightweight containers -------------------------------------------------

@dataclass
class FaceTrack:
    """One visually observed person over the 30s window.

    ``track_id`` is internal (per-tick). Embeddings are averaged over the
    window (§2.1 last sentence).
    """
    track_id: str
    avg_embedding: np.ndarray              # averaged face embedding
    best_bbox: Tuple[int, int, int, int]   # largest clean detection
    quality: float                          # face quality Q^face in [0, 1]
    azimuth_rad: float                      # from bbox centroid
    asd_scores: List[float] = field(default_factory=list)  # per-frame α
    frame_ts: List[float] = field(default_factory=list)    # timestamps for frames that contain this track
    representative_image: Optional[np.ndarray] = None      # for gallery image dump


@dataclass
class DiarizationCluster:
    cluster_id: str
    voice_embedding: Optional[np.ndarray]
    start: float          # seconds from tick start
    end: float            # seconds from tick start
    duration: float
    delta: float          # diarization posterior (speaking confidence)
    audio_bytes: Optional[bytes] = None


@dataclass
class RawObservations:
    """Output of PerceptionEngine.run(). Fed into association.py."""
    face_tracks: List[FaceTrack]
    diar_clusters: List[DiarizationCluster]
    ssl_bins: np.ndarray                  # histogram over azimuth bins (len = num_azimuth_bins)
    ssl_azimuths: List[float]             # raw stream (rad)
    ssl_confidences: List[float]
    num_frames: int
    tick_seconds: float


# ----- tick bundle (what the gRPC layer assembles from the stream) ------------

@dataclass
class TickBundle:
    tick_id: int
    session_id: str
    window_start_ts: float
    robot_heading_rad: float
    audio_pcm: bytes                       # interleaved int16 across `audio_channels`
    audio_sample_rate: int
    audio_channels: int
    frames: List[Tuple[float, np.ndarray]] # (ts, bgr ndarray)
    ssl_events: List[Tuple[float, float, float]]  # (ts, az, conf)


# ----- perception engine ------------------------------------------------------

class PerceptionEngine:
    """Lazy-loaded orchestrator. Instantiate once per server process."""

    def __init__(self, cfg: BraidConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._face_rec = None
        self._diar = None
        self._voice = None
        self._asd_model = None
        self._asd_available = False

    # ---------- model getters ----------

    def _get_face(self):
        if self._face_rec is not None:
            return self._face_rec
        try:
            from ..face_recognition import _FaceRecognition
            # Give BRAID its *own* face db dir so we don't pollute the
            # existing speaker_recognition face_db.
            self._face_rec = _FaceRecognition(
                db_dir=str(Path(self.cfg.gallery_dir) / "face_stage"),
                recognition_threshold=0.55,
            )
            logger.info(f"{C.perception}[perception]{C.r} face recogniser ready")
        except Exception as e:
            logger.exception(f"{C.perception}[perception]{C.r} failed to init face recogniser: %s", e)
            self._face_rec = None
        return self._face_rec

    def _get_diar(self):
        if self._diar is not None:
            return self._diar
        try:
            from ..diarization import _Diarization
            self._diar = _Diarization(max_speakers=self.cfg.max_persons)
            logger.info(f"{C.perception}[perception]{C.r} diarization ready")
        except Exception as e:
            logger.exception(f"{C.perception}[perception]{C.r} diarization unavailable: %s", e)
            self._diar = None
        return self._diar

    def _get_voice(self):
        if self._voice is not None:
            return self._voice
        try:
            from ..speaker_recognition import _SpeakerRecognition
            self._voice = _SpeakerRecognition(model_name="wavlm_ssl")
            logger.info(f"{C.perception}[perception]{C.r} wavlm_ssl voice encoder ready")
        except Exception as e:
            logger.exception(f"{C.perception}[perception]{C.r} voice encoder unavailable: %s", e)
            self._voice = None
        return self._voice

    def _get_asd(self):
        """Light-ASD is in a non-package subdir. We import lazily and
        tolerate its absence (stub returns default α)."""
        if self._asd_model is not None or self._asd_available is False:
            return self._asd_model
        try:
            asd_root = (
                Path(__file__).resolve().parent.parent
                / "active_speaker_detection" / "Light-ASD"
            )
            weight = asd_root / "weight" / "finetuning_TalkSet.model"
            if not weight.exists():
                raise FileNotFoundError(weight)
            if str(asd_root) not in sys.path:
                sys.path.insert(0, str(asd_root))
            import torch  # noqa: F401
            from ASD import ASD as _ASDCls  # type: ignore
            model = _ASDCls()
            model.loadParameters(str(weight))
            model.eval()
            self._asd_model = model
            self._asd_available = True
            logger.info(f"{C.perception}[perception]{C.r} Light-ASD ready")
        except Exception as e:
            logger.warning(f"{C.perception}[perception]{C.r} Light-ASD unavailable (%s); using α=%.2f stub",
                           e, self.cfg.asd_stub_default_alpha)
            self._asd_available = False
        return self._asd_model

    # ---------- public ----------

    def run(self, bundle: TickBundle) -> RawObservations:
        """Full-window pass. Internally:
          1. video: detect/embed/quality per frame, cluster into tracks, avg.
          2. audio: diarize → voice-embed each cluster.
          3. ASD score per (track, frame) subsampled.
          4. SSL stream aggregated into 10° azimuth bins.
        """
        t0 = time.time()
        face_tracks = self._video_pass(bundle)
        diar_clusters = self._audio_pass(bundle)
        ssl_bins, az, conf = self._ssl_pass(bundle)
        self._score_asd_for_tracks(face_tracks, bundle)
        logger.info(
            f"{C.perception}[perception]{C.r} tick=%d faces=%d clusters=%d ssl_events=%d wall=%.2fs",
            bundle.tick_id, len(face_tracks), len(diar_clusters),
            len(bundle.ssl_events), time.time() - t0,
        )
        return RawObservations(
            face_tracks=face_tracks,
            diar_clusters=diar_clusters,
            ssl_bins=ssl_bins,
            ssl_azimuths=az,
            ssl_confidences=conf,
            num_frames=len(bundle.frames),
            tick_seconds=self.cfg.tick_window_seconds,
        )

    # ---------- video ----------

    def _video_pass(self, bundle: TickBundle) -> List[FaceTrack]:
        face_rec = self._get_face()
        if face_rec is None or not bundle.frames:
            return []
        try:
            import cv2  # noqa: F401
        except Exception as e:
            logger.error(f"{C.perception}[perception]{C.r} cv2 unavailable: %s", e)
            return []
        import cv2

        # track-by-embedding: cluster per-frame faces by cosine > 0.5
        track_store: List[Dict[str, Any]] = []
        cam_matrix_cache: Dict[Tuple[int, int], np.ndarray] = {}

        for ts, frame in bundle.frames:
            try:
                faces = face_rec.app.get(frame)
            except Exception:
                continue
            if not faces:
                continue
            h, w = frame.shape[:2]
            if (h, w) not in cam_matrix_cache:
                cam_matrix_cache[(h, w)] = face_rec._get_camera_matrix(frame.shape)
            cam_matrix = cam_matrix_cache[(h, w)]

            for face in faces:
                emb = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                bbox = tuple(int(v) for v in face.bbox.astype(int))
                # find track by cosine
                matched_idx = -1
                best_sim = -1.0
                for idx, tr in enumerate(track_store):
                    sim = float(np.dot(emb, tr["proto"]))
                    if sim > best_sim and sim > 0.5:
                        best_sim = sim
                        matched_idx = idx
                # face quality Q^face (§2.1)
                q = self._face_quality(face, frame, bbox, cam_matrix)
                az = self._bbox_to_azimuth(bbox, w)
                gray_crop = self._extract_asd_crop(frame, bbox)
                if matched_idx < 0:
                    track_store.append({
                        "embs": [emb],
                        "proto": emb.copy(),
                        "bboxes": [bbox],
                        "qualities": [q],
                        "azimuths": [az],
                        "frame_ts": [ts],
                        "gray_crops": [gray_crop] if gray_crop is not None else [],
                        "best_quality_idx": 0,
                        "best_frame": frame,
                        "best_bbox": bbox,
                    })
                else:
                    tr = track_store[matched_idx]
                    tr["embs"].append(emb)
                    # running mean as proto for future matches
                    tr["proto"] = np.mean(tr["embs"], axis=0)
                    tr["proto"] /= (np.linalg.norm(tr["proto"]) + 1e-8)
                    tr["bboxes"].append(bbox)
                    tr["qualities"].append(q)
                    tr["azimuths"].append(az)
                    tr["frame_ts"].append(ts)
                    if gray_crop is not None:
                        tr["gray_crops"].append(gray_crop)
                    if q > tr["qualities"][tr["best_quality_idx"]]:
                        tr["best_quality_idx"] = len(tr["qualities"]) - 1
                        tr["best_frame"] = frame
                        tr["best_bbox"] = bbox

        tracks: List[FaceTrack] = []
        for i, tr in enumerate(track_store):
            if not tr["embs"]:
                continue
            avg = np.mean(tr["embs"], axis=0)
            avg /= (np.linalg.norm(avg) + 1e-8)
            q = float(np.mean(tr["qualities"]))
            az = float(np.mean(tr["azimuths"]))
            # ASD scores populated by _score_asd_for_tracks after audio is known.
            stub_alpha = [self.cfg.asd_stub_default_alpha] * len(tr["frame_ts"])
            ft = FaceTrack(
                track_id=f"t{i}",
                avg_embedding=avg,
                best_bbox=tr["best_bbox"],
                quality=q,
                azimuth_rad=az,
                asd_scores=stub_alpha,
                frame_ts=list(tr["frame_ts"]),
                representative_image=tr["best_frame"],
            )
            ft._gray_crops = tr.get("gray_crops", [])  # type: ignore[attr-defined]
            tracks.append(ft)
            if len(tracks) >= self.cfg.max_persons:
                break
        return tracks

    def _bbox_to_azimuth(self, bbox: Tuple[int, int, int, int], img_w: int) -> float:
        x1, y1, x2, y2 = bbox
        x_center = 0.5 * (x1 + x2)
        # pixel → azimuth per §3.3.3
        return ((x_center - img_w / 2.0) / (img_w / 2.0)) * (self.cfg.camera_hfov / 2.0)

    def _face_quality(self, face, frame: np.ndarray,
                      bbox: Tuple[int, int, int, int], cam_matrix) -> float:
        """§2.1 composite quality:
           Q = min(A/A_min, 1) * exp(-yaw^2 / (2*30^2)) * min(Lap/Lap_min, 1)
        """
        import cv2
        x1, y1, x2, y2 = bbox
        area = max(1, (x2 - x1) * (y2 - y1))
        area_term = min(area / self.cfg.face_area_min_px, 1.0)

        # yaw from head pose via face_recognition helper
        try:
            kps = face.kps.astype(np.float32)
            ok, rvec, _ = cv2.solvePnP(
                np.array([
                    [-30.0, 30.0, -30.0],
                    [30.0, 30.0, -30.0],
                    [0.0, 0.0, 0.0],
                    [-25.0, -30.0, -30.0],
                    [25.0, -30.0, -30.0],
                ], dtype=np.float32),
                kps, cam_matrix, np.zeros((4, 1), dtype=np.float32),
                flags=cv2.SOLVEPNP_EPNP,
            )
            R, _ = cv2.Rodrigues(rvec) if ok else (np.eye(3), None)
            yaw = math.degrees(math.atan2(-R[2, 0], math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)))
        except Exception:
            yaw = 0.0
        yaw_term = math.exp(-(yaw ** 2) / (2.0 * self.cfg.yaw_sigma_deg ** 2))

        # Laplacian sharpness
        try:
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1c:y2c, x1c:x2c]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            lap = self.cfg.lap_min
        lap_term = min(lap / self.cfg.lap_min, 1.0)

        return float(max(0.0, min(1.0, area_term * yaw_term * lap_term)))

    # ---------- audio ----------

    def _audio_mono_bytes(self, bundle: TickBundle) -> bytes:
        """Down-mix 4ch interleaved int16 to mono int16 for single-stream models."""
        if bundle.audio_channels <= 1:
            return bundle.audio_pcm
        try:
            arr = np.frombuffer(bundle.audio_pcm, dtype=np.int16)
            ch = bundle.audio_channels
            n = (arr.shape[0] // ch) * ch
            arr = arr[:n].reshape(-1, ch).astype(np.int32)
            mono = (arr.mean(axis=1)).astype(np.int16)
            return mono.tobytes()
        except Exception as e:
            logger.warning(f"{C.perception}[perception]{C.r} mono downmix failed (%s); using raw bytes", e)
            return bundle.audio_pcm

    def _audio_pass(self, bundle: TickBundle) -> List[DiarizationCluster]:
        mono = self._audio_mono_bytes(bundle)
        sr = bundle.audio_sample_rate or 16000
        diar = self._get_diar()
        voice = self._get_voice()
        if diar is None:
            return []
        try:
            segments = diar.diarize(mono, sample_rate=sr)
        except Exception as e:
            logger.warning(f"{C.perception}[perception]{C.r} diarization failed (%s)", e)
            return []

        # group segments by speaker label → one cluster each
        by_spk: Dict[str, List[dict]] = {}
        for seg in segments:
            by_spk.setdefault(seg["speaker"], []).append(seg)

        clusters: List[DiarizationCluster] = []
        for spk, segs in by_spk.items():
            audio_concat = b"".join(s["audio"] for s in segs)
            total_dur = sum(s["end"] - s["start"] for s in segs)
            start = min(s["start"] for s in segs)
            end = max(s["end"] for s in segs)
            delta = min(1.0, total_dur / max(1e-3, self.cfg.tick_window_seconds))
            emb = None
            if voice is not None and len(audio_concat) > 0:
                try:
                    emb = voice.extract_embedding(audio_concat, sample_rate=sr)
                except Exception as e:
                    logger.warning(f"{C.perception}[perception]{C.r} voice embed failed for %s: %s", spk, e)
                    emb = None
            clusters.append(DiarizationCluster(
                cluster_id=str(spk),
                voice_embedding=emb,
                start=float(start),
                end=float(end),
                duration=float(total_dur),
                delta=float(delta),
                audio_bytes=audio_concat,
            ))
        return clusters

    # ---------- SSL ----------

    def _ssl_pass(self, bundle: TickBundle):
        nb = self.cfg.num_azimuth_bins
        bins = np.zeros(nb, dtype=np.float32)
        az_list: List[float] = []
        conf_list: List[float] = []
        for ts, az, conf in bundle.ssl_events:
            az = (az + math.pi) % (2 * math.pi) - math.pi
            idx = int((az + math.pi) / (2 * math.pi) * nb)
            idx = max(0, min(nb - 1, idx))
            bins[idx] += max(0.0, conf)
            az_list.append(az)
            conf_list.append(conf)
        if bins.sum() > 0:
            bins /= bins.sum()
        return bins, az_list, conf_list

    # ---------- ASD ----------

    def _extract_asd_crop(self, frame: np.ndarray,
                          bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Mirror asd_pipeline.py crop recipe: square face-centred crop,
        resize to 224, take the grayscale centre 112×112 window.
        Returns uint8 (112,112) or None on failure.
        """
        try:
            import cv2
        except Exception:
            return None
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bsize = max(x2 - x1, y2 - y1)
        if bsize <= 0:
            return None
        pad = 0.40  # matches CROP_SCALE in asd_pipeline.py
        half = bsize * (1.0 + pad) * 0.5
        h, w = frame.shape[:2]
        xa = int(max(0, cx - half)); xb = int(min(w, cx + half))
        ya = int(max(0, cy - half)); yb = int(min(h, cy + half))
        if xb - xa < 4 or yb - ya < 4:
            return None
        face = frame[ya:yb, xa:xb]
        try:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if face.ndim == 3 else face
            gray = cv2.resize(gray, (224, 224))
            return gray[56:168, 56:168].copy()   # centre 112×112
        except Exception:
            return None

    def _score_asd_for_tracks(self, tracks: List[FaceTrack],
                              bundle: TickBundle) -> None:
        """Populate ``track.asd_scores`` with real Light-ASD per-frame α.

        Mirrors run_asd_scoring() in /workspace/asd_pipeline.py:
          - MFCC(13) at 100 Hz for the whole mono 16 kHz audio
          - video feat = per-frame grayscale 112×112 crops
          - temporal align on min(audio_secs, video_secs)
          - multi-scale forward over ASD_DURATION_SET, averaged
          - sigmoid(raw) → [0,1] per video frame

        Falls back to the stub (cfg.asd_stub_default_alpha) on any failure.
        """
        if not tracks:
            return
        model = self._get_asd()
        if model is None:
            return

        try:
            import torch
            import cv2  # noqa: F401
            import python_speech_features  # type: ignore
        except Exception as e:
            logger.warning(f"{C.perception}[perception]{C.r} ASD deps missing (%s); keeping stub α", e)
            return

        # Mono 16 kHz int16 → float MFCC.
        sr = bundle.audio_sample_rate or 16000
        mono_bytes = self._audio_mono_bytes(bundle)
        if not mono_bytes:
            return
        audio = np.frombuffer(mono_bytes, dtype=np.int16)
        if audio.size == 0:
            return
        try:
            if sr != 16000:
                # Light-ASD finetuning expects 16 kHz; crude linear resample.
                ratio = 16000.0 / float(sr)
                new_n = int(audio.size * ratio)
                if new_n <= 0:
                    return
                x_old = np.linspace(0, 1, audio.size, endpoint=False)
                x_new = np.linspace(0, 1, new_n, endpoint=False)
                audio = np.interp(x_new, x_old, audio.astype(np.float32)).astype(np.int16)
                sr = 16000
            audio_feat = python_speech_features.mfcc(
                audio, sr, numcep=13, winlen=0.025, winstep=0.010,
            )
        except Exception as e:
            logger.warning(f"{C.perception}[perception]{C.r} MFCC failed (%s); keeping stub α", e)
            return

        device = next(model.parameters()).device if hasattr(model, "parameters") else None
        if device is None:
            try:
                device = next(model.model.parameters()).device
            except Exception:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tick_seconds = max(1e-3, self.cfg.tick_window_seconds)
        DURATIONS = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]   # asd_pipeline.py

        for ft in tracks:
            gray_crops = getattr(ft, "_gray_crops", [])
            if not gray_crops:
                continue
            video_feat = np.stack(gray_crops, axis=0).astype(np.float32)
            T_v = video_feat.shape[0]
            if T_v < 2:
                continue

            # Effective fps inferred from actual timestamps (robust when
            # the capture rate is not exactly 25).
            fps = T_v / tick_seconds
            audio_secs = (audio_feat.shape[0] - audio_feat.shape[0] % 4) / 100.0
            video_secs = T_v / fps
            length = min(audio_secs, video_secs)
            if length <= 0:
                continue

            a_trim = audio_feat[: int(round(length * 100)), :]
            v_trim = video_feat[: int(round(length * fps)), :, :]
            if a_trim.shape[0] == 0 or v_trim.shape[0] == 0:
                continue

            try:
                multi_scores: List[List[float]] = []
                with torch.no_grad():
                    for dur in DURATIONS:
                        n_batches = int(math.ceil(length / dur))
                        scores: List[float] = []
                        for i in range(n_batches):
                            a_slice = a_trim[i * dur * 100:(i + 1) * dur * 100, :]
                            v_slice = v_trim[i * dur * int(round(fps)):(i + 1) * dur * int(round(fps)), :, :]
                            if a_slice.shape[0] == 0 or v_slice.shape[0] == 0:
                                continue
                            inputA = torch.as_tensor(a_slice, dtype=torch.float32, device=device).unsqueeze(0)
                            inputV = torch.as_tensor(v_slice, dtype=torch.float32, device=device).unsqueeze(0)
                            embedA = model.model.forward_audio_frontend(inputA)
                            embedV = model.model.forward_visual_frontend(inputV)
                            out = model.model.forward_audio_visual_backend(embedA, embedV)
                            raw = model.lossAV.forward(out, labels=None)
                            # lossAV returns either list-like per-frame scores or tensor
                            if hasattr(raw, "detach"):
                                raw = raw.detach().float().cpu().numpy().reshape(-1).tolist()
                            else:
                                raw = list(raw)
                            scores.extend(raw)
                        if scores:
                            # pad/trim each scale to T_v for safe averaging
                            if len(scores) < T_v:
                                scores = scores + [scores[-1]] * (T_v - len(scores))
                            else:
                                scores = scores[:T_v]
                            multi_scores.append(scores)

                if not multi_scores:
                    continue
                mean_raw = np.mean(np.asarray(multi_scores, dtype=np.float32), axis=0)
                # Map raw score to α ∈ [0,1]. lossAV outputs are logit-like
                # around 0 at the speaking/silent boundary → sigmoid is the
                # natural calibration.
                alpha = 1.0 / (1.0 + np.exp(-mean_raw))
                ft.asd_scores = [float(x) for x in alpha.tolist()]
            except Exception as e:
                logger.warning(f"{C.perception}[perception]{C.r} Light-ASD forward failed on %s (%s); "
                               "keeping stub α for this track", ft.track_id, e)
                continue

        logger.info(f"{C.perception}[perception]{C.r} ASD scored %d tracks", len(tracks))
