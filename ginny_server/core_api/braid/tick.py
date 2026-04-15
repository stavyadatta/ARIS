"""Single-tick orchestrator (Blueprint §6).

``run_tick(bundle, engine, gallery, session_state, cfg)`` is the main glue:
perception → association → temporal re-associate → posterior → decision →
action_policy → gallery writes. Returns a BraidTickResult-ready dataclass.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .action_policy import BraidAction, select_action
from .association import PersonObservation, associate
from .config import BraidConfig
from .decision import BraidDecision, DecisionState, decide
from .gallery import BraidGallery
from .log_style import C, state_color
from .perception import PerceptionEngine, TickBundle
from .posterior import IdentityPosterior, PosteriorComputer
from .temporal import (
    SessionState,
    commit_memory,
    identity_prior_for,
    location_prior_for,
    reassociate,
)

logger = logging.getLogger("braid")


@dataclass
class PersonTickResult:
    stable_id: str
    observation: PersonObservation
    posterior: IdentityPosterior
    decision: BraidDecision
    prev_state: str


@dataclass
class BraidTickResult:
    tick_id: int
    session_id: str
    persons: List[PersonTickResult] = field(default_factory=list)
    action: BraidAction = field(default_factory=lambda: BraidAction("STAY", 0.0, "noop", ""))
    tick_wall_seconds: float = 0.0


def run_tick(
    bundle: TickBundle,
    engine: PerceptionEngine,
    gallery: BraidGallery,
    session_state: SessionState,
    cfg: BraidConfig,
) -> BraidTickResult:
    t0 = time.time()
    logger.info(f"{C.tick}{C.bold}========== [tick] START tick=%d session=%s "
                f"heading=%.2frad prior_memories=%d gallery=%d =========={C.r}",
                bundle.tick_id, bundle.session_id, bundle.robot_heading_rad,
                len(session_state.memories), len(gallery.entries()))

    # 1. Perception.
    logger.info(f"{C.tick}[tick]{C.r} phase=1 perception — running face/ASD/diar/voice/SSL pipelines")
    t_p = time.time()
    obs = engine.run(bundle)
    logger.info(f"{C.tick}[tick]{C.r} perception done in %.2fs: face_tracks=%d diar_clusters=%d "
                "ssl_events=%d",
                time.time() - t_p, len(obs.face_tracks),
                len(obs.diar_clusters), len(obs.ssl_azimuths))

    # 2. Audio-visual association (+ phantoms).
    logger.info(f"{C.tick}[tick]{C.r} phase=2 association — bridging faces↔clusters (tau_bridge=%.2f)",
                cfg.tau_bridge)
    persons = associate(obs, cfg)
    vis = sum(1 for p in persons if p.visible)
    logger.info(f"{C.tick}[tick]{C.r} association done: persons=%d (visible=%d phantom=%d)",
                len(persons), vis, len(persons) - vis)

    # 3. Temporal re-association with prior memory.
    logger.info(f"{C.tick}[tick]{C.r} phase=3 temporal reassoc — face_cos>%.2f or voice_cos>%.2f",
                cfg.reassoc_face_cos, cfg.reassoc_voice_cos)
    pairs = reassociate(session_state, persons, cfg)
    matched = sum(1 for _, m in pairs if m is not None)
    logger.info(f"{C.tick}[tick]{C.r} reassoc done: matched=%d new=%d", matched, len(pairs) - matched)

    # 4. Posterior + decision per person.
    logger.info(f"{C.tick}[tick]{C.r} phase=4 posterior+decision — gallery size M=%d",
                len(gallery.entries()))
    computer = PosteriorComputer(cfg)
    results: List[PersonTickResult] = []
    for po, prev_mem in pairs:
        id_prior = identity_prior_for(prev_mem, cfg.p_new)
        loc_prior = location_prior_for(
            prev_mem,
            prev_heading=session_state.last_heading_rad,
            new_heading=bundle.robot_heading_rad,
        )
        post = computer.compute(po, gallery,
                                identity_prior=id_prior,
                                location_prior_az=loc_prior)
        dec = decide(po, post, cfg)
        prev_state = prev_mem.last_state if prev_mem else "NEW"

        logger.info(
            f"{C.tick}[tick]{C.r} person=%s posterior: p_best=%.3f p_unk=%.3f margin=%.3f "
            "H=%.2f Q_gate=%.2f j_best=%s mod_agree=%s",
            po.person_id, post.p_best, post.p_unk, post.margin, post.entropy,
            post.Q_gate, post.j_best or "-", post.modality_agreement,
        )

        # 5. Gallery writes.
        if dec.state == DecisionState.ENROL:
            logger.info(f"{C.tick}[tick]{C.r} phase=5 gallery ENROL for person=%s", po.person_id)
            gallery.enrol(
                face_emb=po.face_emb,
                voice_emb=po.voice_emb,
                face_quality=po.face_quality,
                representative_image=po.representative_image,
            )
        elif dec.state == DecisionState.RECOGNISE and dec.identity:
            logger.info(f"{C.tick}[tick]{C.r} phase=5 gallery UPDATE_EMA person=%s identity=%s α=%.2f",
                        po.person_id, dec.identity, cfg.gallery_ema_alpha)
            gallery.update_ema(
                dec.identity,
                face_emb=po.face_emb if po.visible else None,
                voice_emb=po.voice_emb,
                alpha=cfg.gallery_ema_alpha,
                face_quality=po.face_quality if po.visible else None,
            )

        # 6. Commit memory for next tick (after decision, so state is accurate).
        sid = commit_memory(session_state, po, post, dec.state.value,
                            bundle.tick_id, prev_mem)

        results.append(PersonTickResult(
            stable_id=sid, observation=po, posterior=post,
            decision=dec, prev_state=prev_state,
        ))

        # 7. Structured per-person log line (spec requirement).
        sc = state_color(dec.state.value)
        logger.info(
            f"{C.bold}person=%s{C.r} decision=%s→{sc}%s{C.r} p_best=%.2f p_unk=%.2f "
            "margin=%.2f Q_gate=%.2f H=%.2f mod_agree=%s identity=%s reason=%s",
            sid, prev_state, dec.state.value, post.p_best, post.p_unk,
            post.margin, post.Q_gate, post.entropy, post.modality_agreement,
            dec.identity or "-", dec.reason,
        )

    # 8. Action selection.
    logger.info(f"{C.tick}[tick]{C.r} phase=6 action_policy — %d persons considered", len(results))
    action = select_action(
        [(r.observation, r.posterior, r.decision) for r in results],
        robot_heading_rad=bundle.robot_heading_rad,
        cfg=cfg,
    )
    logger.info(
        f"{C.action}action=%s{C.r} magnitude=%.2f%s reason=%s target_person=%s",
        action.type.lower(),
        math.degrees(action.magnitude) if action.type.startswith("ROTATE")
        else action.magnitude,
        "°" if action.type.startswith("ROTATE") else ("m" if action.type == "MOVE_FORWARD" else ""),
        action.reason, action.target_person_id or "-",
    )

    session_state.last_heading_rad = bundle.robot_heading_rad
    session_state.last_tick_id = bundle.tick_id

    logger.info(f"{C.tick}{C.bold}========== [tick] END tick=%d wall=%.2fs "
                f"persons=%d action=%s =========={C.r}",
                bundle.tick_id, time.time() - t0, len(results), action.type)

    return BraidTickResult(
        tick_id=bundle.tick_id,
        session_id=bundle.session_id,
        persons=results,
        action=action,
        tick_wall_seconds=time.time() - t0,
    )
