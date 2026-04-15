"""Five-state decision rules (Blueprint §3.5).

Priority order (per spec): RECOGNISE → CONFIRM → ENROL → EXPLORE → UNKNOWN.
Returns a ``BraidDecision`` dataclass that downstream code (action_policy,
gallery, grpc_handle) consumes.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Optional

from .association import PersonObservation
from .config import BraidConfig
from .log_style import C, state_color
from .posterior import IdentityPosterior

logger = logging.getLogger("braid")


class DecisionState(str, enum.Enum):
    RECOGNISE = "RECOGNISE"
    CONFIRM = "CONFIRM"
    ENROL = "ENROL"
    EXPLORE = "EXPLORE"
    UNKNOWN = "UNKNOWN"


@dataclass
class BraidDecision:
    person_id: str
    state: DecisionState
    identity: Optional[str]           # gallery id on RECOGNISE, else None
    confidence: float                 # p_best when applicable, else 0
    reason: str                       # "no_signal", "weak_match", "not_visible", ...
    target_azimuth_rad: Optional[float]  # for EXPLORE targeting


def decide(person: PersonObservation,
           post: IdentityPosterior,
           cfg: BraidConfig) -> BraidDecision:
    has_face = person.visible and person.face_quality > cfg.Q_min \
        and person.face_emb is not None
    has_voice = (person.voice_emb is not None) and (person.diar_delta > 0.3)
    has_any = has_face or has_voice
    logger.info(f"{C.decision}[decision]{C.r} person=%s has_face=%s has_voice=%s visible=%s "
                "face_q=%.2f diar_delta=%.2f",
                person.person_id, has_face, has_voice, person.visible,
                person.face_quality, person.diar_delta)

    if not has_any:
        logger.info(f"{C.decision}[decision]{C.r} person=%s → EXPLORE(no_signal)", person.person_id)
        target_az = person.ssl_azimuth_rad if person.ssl_azimuth_rad is not None \
            else person.face_azimuth_rad
        return BraidDecision(
            person_id=person.person_id,
            state=DecisionState.EXPLORE,
            identity=None,
            confidence=0.0,
            reason="no_signal",
            target_azimuth_rad=target_az,
        )

    p_best = post.p_best
    p_unk = post.p_unk
    Q_gate = post.Q_gate
    H = post.entropy
    j_best = post.j_best

    # ---- 1. RECOGNISE (§3.5 Decision 1) --------------------------------------
    if j_best is not None and p_best > cfg.tau_recog and Q_gate > cfg.Q_recog:
        if has_face and has_voice and post.modality_agreement:
            logger.info(f"{C.decision}[decision]{C.r} person=%s → RECOGNISE(fused_match id=%s)",
                        person.person_id, j_best)
            return BraidDecision(person.person_id, DecisionState.RECOGNISE,
                                 j_best, p_best, "fused_match", None)
        if has_face and person.face_quality > 0.7:
            logger.info(f"{C.decision}[decision]{C.r} person=%s → RECOGNISE(strong_face id=%s)",
                        person.person_id, j_best)
            return BraidDecision(person.person_id, DecisionState.RECOGNISE,
                                 j_best, p_best, "strong_face", None)
        if has_voice and person.diar_delta > 0.7:
            logger.info(f"{C.decision}[decision]{C.r} person=%s → RECOGNISE(strong_voice id=%s)",
                        person.person_id, j_best)
            return BraidDecision(person.person_id, DecisionState.RECOGNISE,
                                 j_best, p_best, "strong_voice", None)

    # ---- 2. CONFIRM ----------------------------------------------------------
    if j_best is not None and p_best > cfg.tau_confirm and Q_gate > cfg.Q_confirm:
        if (not post.modality_agreement) and has_face and has_voice:
            logger.info(f"{C.decision}[decision]{C.r} person=%s → CONFIRM(modality_disagreement)",
                        person.person_id)
            return BraidDecision(person.person_id, DecisionState.CONFIRM,
                                 j_best, p_best, "modality_disagreement",
                                 person.face_azimuth_rad)
        if p_best <= cfg.tau_recog:
            logger.info(f"{C.decision}[decision]{C.r} person=%s → CONFIRM(weak_match p_best=%.2f)",
                        person.person_id, p_best)
            return BraidDecision(person.person_id, DecisionState.CONFIRM,
                                 j_best, p_best, "weak_match",
                                 person.face_azimuth_rad)

    # ---- 3. ENROL ------------------------------------------------------------
    if p_unk > cfg.tau_enrol_unk:
        if has_face and person.face_quality > cfg.Q_enrol:
            reason = "enrol_face_voice" if has_voice else "enrol_face_only"
            logger.info(f"{C.decision}[decision]{C.r} person=%s → ENROL(%s p_unk=%.2f q=%.2f)",
                        person.person_id, reason, p_unk, person.face_quality)
            return BraidDecision(person.person_id, DecisionState.ENROL,
                                 None, p_unk, reason, None)

    # ---- 4. EXPLORE ----------------------------------------------------------
    if H > cfg.H_explore:
        if not person.visible:
            logger.info(f"{C.decision}[decision]{C.r} person=%s → EXPLORE(not_visible az=%s)",
                        person.person_id,
                        f"{person.ssl_azimuth_rad:.2f}" if person.ssl_azimuth_rad else "None")
            return BraidDecision(person.person_id, DecisionState.EXPLORE,
                                 None, p_best, "not_visible",
                                 person.ssl_azimuth_rad)
        if person.face_quality < cfg.Q_min_enrol:
            logger.info(f"{C.decision}[decision]{C.r} person=%s → EXPLORE(low_quality_face q=%.2f)",
                        person.person_id, person.face_quality)
            return BraidDecision(person.person_id, DecisionState.EXPLORE,
                                 None, p_best, "low_quality_face",
                                 person.face_azimuth_rad)
        if not has_voice:
            logger.info(f"{C.decision}[decision]{C.r} person=%s → EXPLORE(no_voice_yet)",
                        person.person_id)
            return BraidDecision(person.person_id, DecisionState.EXPLORE,
                                 None, p_best, "no_voice_yet",
                                 person.face_azimuth_rad)

    # ---- 5. UNKNOWN ----------------------------------------------------------
    logger.info(f"{C.decision}[decision]{C.r} person=%s → UNKNOWN(insufficient_data H=%.2f p_best=%.2f)",
                person.person_id, H, p_best)
    return BraidDecision(person.person_id, DecisionState.UNKNOWN,
                         None, p_best, "insufficient_data", None)
