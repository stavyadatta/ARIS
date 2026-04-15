"""Temporal propagation between ticks (Blueprint §5).

Responsibilities:
  * Carry identity posterior forward as next-tick prior (§5.2).
  * Propagate location belief with Gaussian drift (§5.1).
  * Compensate for robot body rotation between ticks (§5.1 last eq).
  * Re-associate this tick's observations with prior hypotheses by face/voice
    cosine > 0.5 (§5.3).

A ``SessionState`` object is held per gRPC session_id.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .association import PersonObservation
from .config import BraidConfig
from .log_style import C
from .posterior import IdentityPosterior

logger = logging.getLogger("braid")


@dataclass
class PersonMemory:
    """Last-tick info for one person hypothesis."""
    stable_id: str
    face_emb: Optional[np.ndarray]
    voice_emb: Optional[np.ndarray]
    last_azimuth_rad: Optional[float]
    identity_prior: Dict[str, float]  # ID → prob (incl. "__unk__")
    last_state: str
    last_tick_id: int


@dataclass
class SessionState:
    session_id: str
    last_heading_rad: float = 0.0
    last_tick_id: int = -1
    memories: Dict[str, PersonMemory] = field(default_factory=dict)
    next_stable: int = 1


def _cos(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a.reshape(-1), b.reshape(-1)) / (na * nb))


def reassociate(
    state: SessionState,
    persons: List[PersonObservation],
    cfg: BraidConfig,
) -> List[Tuple[PersonObservation, Optional[PersonMemory]]]:
    """Map each current-tick observation to a prior memory (or None if new).

    Cosine > cfg.reassoc_face_cos on face, else > cfg.reassoc_voice_cos on voice.
    Greedy best-match; memories are consumed at most once per tick.
    """
    logger.info(f"{C.temporal}[temporal]{C.r} ENTER reassoc: %d incoming persons vs %d memories",
                len(persons), len(state.memories))
    claimed: set = set()
    pairs: List[Tuple[PersonObservation, Optional[PersonMemory]]] = []
    for po in persons:
        best_mem: Optional[PersonMemory] = None
        best_score = 0.0
        for sid, mem in state.memories.items():
            if sid in claimed:
                continue
            face_sim = _cos(po.face_emb, mem.face_emb)
            voice_sim = _cos(po.voice_emb, mem.voice_emb)
            score = max(face_sim, voice_sim * 0.9)  # slight face preference
            gate = (face_sim > cfg.reassoc_face_cos) or (voice_sim > cfg.reassoc_voice_cos)
            if gate and score > best_score:
                best_score = score
                best_mem = mem
        if best_mem is not None:
            claimed.add(best_mem.stable_id)
            logger.info(f"{C.temporal}[temporal]{C.r} reassoc %s → %s score=%.2f",
                        po.person_id, best_mem.stable_id, best_score)
        pairs.append((po, best_mem))
    return pairs


def location_prior_for(mem: Optional[PersonMemory],
                       prev_heading: float, new_heading: float) -> Optional[float]:
    """Propagate last azimuth from previous robot frame to the new one (§5.1)."""
    if mem is None or mem.last_azimuth_rad is None:
        return None
    delta = new_heading - prev_heading
    az = mem.last_azimuth_rad - delta
    # wrap
    return ((az + math.pi) % (2 * math.pi)) - math.pi


def identity_prior_for(mem: Optional[PersonMemory],
                       default_p_new: float) -> Optional[Dict[str, float]]:
    if mem is None:
        return None
    return dict(mem.identity_prior)


def commit_memory(
    state: SessionState,
    person: PersonObservation,
    post: IdentityPosterior,
    decision_state_name: str,
    tick_id: int,
    prev_mem: Optional[PersonMemory],
) -> str:
    """Update or create a PersonMemory for this hypothesis."""
    # Stable id: prefer matched mem's id, else allocate new
    if prev_mem is not None:
        sid = prev_mem.stable_id
    else:
        sid = f"s_{state.next_stable}"
        state.next_stable += 1

    # Identity prior = this tick's posterior (carries over).
    id_prior: Dict[str, float] = {}
    for iid, p in zip(post.ids, post.probs):
        id_prior[iid] = float(p)

    # EMA on stored embeddings so memory doesn't jitter tick-to-tick.
    def _ema(prev: Optional[np.ndarray], new: Optional[np.ndarray], alpha: float = 0.3):
        if new is None:
            return prev
        if prev is None:
            return new.astype(np.float32).copy()
        return ((1 - alpha) * prev + alpha * new.astype(np.float32)).astype(np.float32)

    state.memories[sid] = PersonMemory(
        stable_id=sid,
        face_emb=_ema(prev_mem.face_emb if prev_mem else None, person.face_emb),
        voice_emb=_ema(prev_mem.voice_emb if prev_mem else None, person.voice_emb),
        last_azimuth_rad=(person.face_azimuth_rad if person.visible
                          else person.ssl_azimuth_rad),
        identity_prior=id_prior,
        last_state=decision_state_name,
        last_tick_id=tick_id,
    )
    return sid
