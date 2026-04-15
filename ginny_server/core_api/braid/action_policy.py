"""Action policy (Blueprint §4.1 / §4.3).

Greedy highest-uncertainty selection; ties broken by higher entropy then by
smaller absolute azimuth magnitude (closer to forward). Emits exactly one
BraidAction per tick.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .association import PersonObservation
from .config import BraidConfig
from .decision import BraidDecision, DecisionState
from .posterior import IdentityPosterior

logger = logging.getLogger("braid")


@dataclass
class BraidAction:
    type: str                # "ROTATE_LEFT" | "ROTATE_RIGHT" | "MOVE_FORWARD" | "STAY"
    magnitude: float         # radians for rotate_*, meters for move_forward, 0 for STAY
    reason: str
    target_person_id: str = ""


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def select_action(
    persons: List[Tuple[PersonObservation, IdentityPosterior, BraidDecision]],
    robot_heading_rad: float,
    cfg: BraidConfig,
) -> BraidAction:
    """Pick the single action for this tick."""
    candidates = [p for p in persons
                  if p[2].state in (DecisionState.EXPLORE, DecisionState.CONFIRM)]
    logger.info("[action] %d/%d persons need attention (EXPLORE/CONFIRM)",
                len(candidates), len(persons))
    if not candidates:
        logger.info("[action] no candidates → STAY(all_resolved)")
        return BraidAction("STAY", 0.0, "all_resolved", "")

    # Greedy: highest entropy (uncertainty). Tiebreak: smaller |azimuth|.
    def _key(item):
        _, post, dec = item
        az = dec.target_azimuth_rad if dec.target_azimuth_rad is not None else 0.0
        return (-post.entropy, abs(_wrap(az)))

    candidates.sort(key=_key)
    person, post, dec = candidates[0]
    logger.info("[action] selected target=%s state=%s reason=%s H=%.2f",
                person.person_id, dec.state.value, dec.reason, post.entropy)

    if dec.state == DecisionState.EXPLORE:
        if dec.reason == "not_visible":
            az = dec.target_azimuth_rad if dec.target_azimuth_rad is not None else 0.0
            delta = _wrap(az - robot_heading_rad)
            if abs(delta) > cfg.rotate_min_rad:
                atype = "ROTATE_LEFT" if delta > 0 else "ROTATE_RIGHT"
                return BraidAction(atype, abs(delta), "phantom_speaker",
                                   person.person_id)
            return BraidAction("STAY", 0.0, "phantom_within_deadband",
                               person.person_id)
        if dec.reason == "low_quality_face":
            return BraidAction("MOVE_FORWARD", cfg.move_forward_m,
                               "improve_face_crop", person.person_id)
        if dec.reason == "no_voice_yet":
            return BraidAction("STAY", 0.0, "wait_for_speech", person.person_id)
        # no_signal fallback: rotate toward target azimuth if any
        if dec.target_azimuth_rad is not None:
            delta = _wrap(dec.target_azimuth_rad - robot_heading_rad)
            if abs(delta) > cfg.rotate_min_rad:
                atype = "ROTATE_LEFT" if delta > 0 else "ROTATE_RIGHT"
                return BraidAction(atype, abs(delta), "no_signal_rotate",
                                   person.person_id)
        return BraidAction("STAY", 0.0, "no_signal_stay", person.person_id)

    # CONFIRM
    if dec.state == DecisionState.CONFIRM:
        if dec.reason == "modality_disagreement":
            az = dec.target_azimuth_rad if dec.target_azimuth_rad is not None else 0.0
            delta = _wrap(az - robot_heading_rad)
            atype = "ROTATE_LEFT" if delta >= 0 else "ROTATE_RIGHT"
            return BraidAction(atype, cfg.rotate_nudge_rad, "confirm_angle_nudge",
                               person.person_id)
        return BraidAction("STAY", 0.0, "confirm_accumulate", person.person_id)

    return BraidAction("STAY", 0.0, "noop", "")
