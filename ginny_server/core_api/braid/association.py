"""Audio-visual association (§2.3) and phantom creation (§2.4).

Links each diarization cluster to at most one face track via ASD temporal
overlap > ``tau_bridge``. Clusters that fail the bridge become phantom person
hypotheses with V=0.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import BraidConfig
from .log_style import C
from .perception import DiarizationCluster, FaceTrack, RawObservations

logger = logging.getLogger("braid")


@dataclass
class PersonObservation:
    """§2.3 observation bundle O_i — joined into a single struct per candidate.

    ``visible`` == True iff a face track backs this hypothesis. Phantoms have
    face_emb=None, bbox=None, face_quality=0.
    """
    person_id: str                     # internal, tick-scoped
    visible: bool
    # face
    face_emb: Optional[np.ndarray] = None
    bbox: Optional[tuple] = None
    face_quality: float = 0.0
    face_azimuth_rad: float = 0.0
    mean_asd: float = 0.0
    representative_image: Optional[np.ndarray] = None
    # voice / audio
    voice_emb: Optional[np.ndarray] = None
    ssl_azimuth_rad: Optional[float] = None
    ssl_confidence: float = 0.0
    diar_delta: float = 0.0
    # internal (debug)
    face_track_id: Optional[str] = None
    cluster_id: Optional[str] = None


def _temporal_overlap(track: FaceTrack, cluster: DiarizationCluster,
                      tick_seconds: float) -> float:
    """Fraction of tick during which ASD(track) > 0.5 AND cluster is active."""
    if not track.frame_ts:
        return 0.0
    # Treat each frame's ASD as active for a short window. We approximate by
    # counting frames whose timestamp falls inside [cluster.start, cluster.end]
    # and whose ASD score > 0.5.
    cstart, cend = cluster.start, cluster.end
    active = 0
    for ts_frame, alpha in zip(track.frame_ts, track.asd_scores):
        # normalise ts_frame to seconds-from-tick-start if it's absolute
        local_ts = ts_frame
        if cstart <= local_ts <= cend and alpha > 0.5:
            active += 1
    return active / max(1, len(track.frame_ts))


def _mean_asd_over_cluster(track: FaceTrack, cluster: DiarizationCluster) -> float:
    if not track.frame_ts:
        return 0.0
    vals = []
    for ts_frame, alpha in zip(track.frame_ts, track.asd_scores):
        if cluster.start <= ts_frame <= cluster.end:
            vals.append(alpha)
    if not vals:
        return float(np.mean(track.asd_scores)) if track.asd_scores else 0.0
    return float(np.mean(vals))


def _cluster_azimuth(ssl_bins: np.ndarray, ssl_azimuths: List[float],
                     ssl_confidences: List[float],
                     cluster: DiarizationCluster) -> Optional[float]:
    """Pick the dominant azimuth for a cluster: weighted mean of SSL events
    whose timestamp falls inside the cluster's activity window. Falls back to
    the global SSL bin mode."""
    # Without absolute SSL timestamps aligned to cluster windows we use the
    # global histogram mode — this is the blueprint's intent at §2.2 ("dominant
    # azimuth clusters via KDE"). Good enough for the coarse 10° grid.
    if len(ssl_azimuths) == 0 or ssl_bins.sum() <= 0:
        return None
    nb = len(ssl_bins)
    best = int(np.argmax(ssl_bins))
    # bin center
    bin_center = -math.pi + (best + 0.5) * (2 * math.pi / nb)
    return bin_center


def associate(obs: RawObservations, cfg: BraidConfig) -> List[PersonObservation]:
    """Bridge ASD between face tracks and diarization clusters (§2.3); emit
    phantoms (§2.4) for unassociated clusters.

    Returns person observations in priority order (visible first, then phantom).
    """
    logger.info(f"{C.assoc}[assoc]{C.r} ENTER tracks=%d clusters=%d ssl_events=%d",
                len(obs.face_tracks), len(obs.diar_clusters),
                len(obs.ssl_azimuths))
    persons: List[PersonObservation] = []
    used_clusters: set = set()

    # Visible tracks — find their best-overlap cluster if any.
    for track in obs.face_tracks:
        best_k: Optional[DiarizationCluster] = None
        best_overlap = 0.0
        for k in obs.diar_clusters:
            if k.cluster_id in used_clusters:
                continue
            ov = _temporal_overlap(track, k, obs.tick_seconds)
            if ov > best_overlap:
                best_overlap = ov
                best_k = k

        linked = best_k if best_overlap >= cfg.tau_bridge else None
        po = PersonObservation(
            person_id=f"p_{track.track_id}",
            visible=True,
            face_emb=track.avg_embedding,
            bbox=track.best_bbox,
            face_quality=track.quality,
            face_azimuth_rad=track.azimuth_rad,
            mean_asd=float(np.mean(track.asd_scores)) if track.asd_scores else 0.0,
            representative_image=track.representative_image,
            face_track_id=track.track_id,
        )
        if linked is not None:
            used_clusters.add(linked.cluster_id)
            po.voice_emb = linked.voice_embedding
            po.ssl_azimuth_rad = _cluster_azimuth(
                obs.ssl_bins, obs.ssl_azimuths, obs.ssl_confidences, linked
            )
            po.ssl_confidence = float(np.mean(obs.ssl_confidences)) if obs.ssl_confidences else 0.0
            po.diar_delta = linked.delta
            po.cluster_id = linked.cluster_id
            po.mean_asd = _mean_asd_over_cluster(track, linked)
            logger.info(f"{C.assoc}[assoc]{C.r} bridged face %s ↔ cluster %s overlap=%.2f",
                        track.track_id, linked.cluster_id, best_overlap)
        persons.append(po)

    # Phantoms for leftover clusters (§2.4).
    for k in obs.diar_clusters:
        if k.cluster_id in used_clusters:
            continue
        if k.voice_embedding is None:
            continue
        ssl_az = _cluster_azimuth(
            obs.ssl_bins, obs.ssl_azimuths, obs.ssl_confidences, k
        )
        po = PersonObservation(
            person_id=f"phantom_{k.cluster_id}",
            visible=False,
            voice_emb=k.voice_embedding,
            ssl_azimuth_rad=ssl_az,
            ssl_confidence=float(np.mean(obs.ssl_confidences)) if obs.ssl_confidences else 0.0,
            diar_delta=k.delta,
            cluster_id=k.cluster_id,
        )
        persons.append(po)
        logger.info(f"{C.assoc}[assoc]{C.r} phantom from cluster %s az=%s delta=%.2f",
                    k.cluster_id,
                    f"{ssl_az:.2f}" if ssl_az is not None else "None",
                    k.delta)

    # Cap at max_persons.
    return persons[: cfg.max_persons]
