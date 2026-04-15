"""gRPC servicer for ``BraidService.RunTick``.

Streams in a 30s tick bundle (audio chunks + frame JPEGs + SSL events +
meta), runs the BRAID pipeline, returns one BraidTickResult.

Thread-safety: per-session ``SessionState`` objects are stored in a dict
protected by a lock so concurrent ticks (different session_ids) don't stomp
each other.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from typing import Dict, List, Optional

import grpc

from grpc_communication import grpc_pb2 as pb2
from grpc_communication import grpc_pb2_grpc as pb2_grpc

from .config import load_config
from .decision import DecisionState
from .gallery import BraidGallery
from .log_style import C
from .perception import PerceptionEngine, TickBundle
from .temporal import SessionState
from .tick import run_tick

logger = logging.getLogger("braid")


_DECISION_ENUM = {
    DecisionState.RECOGNISE: pb2.DecisionState.RECOGNISE,
    DecisionState.CONFIRM:   pb2.DecisionState.CONFIRM,
    DecisionState.ENROL:     pb2.DecisionState.ENROL,
    DecisionState.EXPLORE:   pb2.DecisionState.EXPLORE,
    DecisionState.UNKNOWN:   pb2.DecisionState.UNKNOWN,
}

_ACTION_ENUM = {
    "STAY":         pb2.ActionType.STAY,
    "ROTATE_LEFT":  pb2.ActionType.ROTATE_LEFT,
    "ROTATE_RIGHT": pb2.ActionType.ROTATE_RIGHT,
    "MOVE_FORWARD": pb2.ActionType.MOVE_FORWARD,
}


class BraidServiceServicer(pb2_grpc.BraidServiceServicer):
    def __init__(self, cfg=None, engine: Optional[PerceptionEngine] = None,
                 gallery: Optional[BraidGallery] = None):
        self.cfg = cfg or load_config()
        self.engine = engine or PerceptionEngine(self.cfg)
        self.gallery = gallery or BraidGallery(self.cfg.gallery_dir)
        self._sessions: Dict[str, SessionState] = {}
        self._sess_lock = threading.Lock()

    # ------ helpers ---------------------------------------------------------

    def _get_session(self, session_id: str) -> SessionState:
        with self._sess_lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState(session_id=session_id)
            return self._sessions[session_id]

    def _assemble_bundle(self, request_iterator) -> TickBundle:
        """Drain the client stream into a TickBundle."""
        tick_id: int = 0
        session_id: str = "default"
        heading: float = 0.0
        ws_ts: float = time.time()
        sr: int = 16000
        ch: int = 4
        audio_chunks: List[bytes] = []
        frames = []
        ssl_events = []
        for chunk in request_iterator:
            if chunk.tick_id:
                tick_id = int(chunk.tick_id)
            if chunk.session_id:
                session_id = chunk.session_id
            if chunk.robot_heading_rad:
                heading = float(chunk.robot_heading_rad)

            payload = chunk.WhichOneof("payload")
            if payload == "meta":
                m = chunk.meta
                if m.session_id: session_id = m.session_id
                if m.tick_id: tick_id = int(m.tick_id)
                if m.window_start_ts: ws_ts = float(m.window_start_ts)
                if m.robot_heading_rad: heading = float(m.robot_heading_rad)
                if m.audio_sample_rate: sr = int(m.audio_sample_rate)
                if m.audio_num_channels: ch = int(m.audio_num_channels)
            elif payload == "audio_chunk":
                a = chunk.audio_chunk
                audio_chunks.append(a.pcm)
                if a.sample_rate: sr = int(a.sample_rate)
                if a.channels: ch = int(a.channels)
            elif payload == "frame_chunk":
                f = chunk.frame_chunk
                try:
                    import numpy as np
                    import cv2
                    arr = np.frombuffer(f.jpeg, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        frames.append((float(f.ts), img))
                except Exception as e:
                    logger.warning(f"{C.grpc}[grpc]{C.r} frame decode failed: %s", e)
            elif payload == "ssl_event":
                s = chunk.ssl_event
                ssl_events.append((float(s.ts), float(s.azimuth_rad),
                                   float(s.confidence)))

        return TickBundle(
            tick_id=tick_id,
            session_id=session_id,
            window_start_ts=ws_ts,
            robot_heading_rad=heading,
            audio_pcm=b"".join(audio_chunks),
            audio_sample_rate=sr,
            audio_channels=ch,
            frames=frames,
            ssl_events=ssl_events,
        )

    def _build_result(self, tick_res) -> pb2.BraidTickResult:
        out = pb2.BraidTickResult(
            tick_id=tick_res.tick_id,
            session_id=tick_res.session_id,
            tick_wall_seconds=float(tick_res.tick_wall_seconds),
        )
        for r in tick_res.persons:
            pd = pb2.PersonDecision(
                person_id=r.stable_id,
                state=_DECISION_ENUM.get(r.decision.state, pb2.DecisionState.UNKNOWN),
                p_best=float(r.posterior.p_best),
                p_unk=float(r.posterior.p_unk),
                margin=float(r.posterior.margin),
                entropy=float(r.posterior.entropy),
                identity=r.decision.identity or "",
                modality_agreement=bool(r.posterior.modality_agreement),
                face_quality=float(r.observation.face_quality),
                ssl_azimuth_rad=float(r.observation.ssl_azimuth_rad or 0.0),
                reason=r.decision.reason,
            )
            out.persons.append(pd)
        a = tick_res.action
        out.action.CopyFrom(pb2.BraidAction(
            type=_ACTION_ENUM.get(a.type, pb2.ActionType.STAY),
            magnitude=float(a.magnitude),
            reason=a.reason,
            target_person_id=a.target_person_id or "",
        ))
        return out

    # ------ RPC -------------------------------------------------------------

    def RunTick(self, request_iterator, context):
        t0 = time.time()
        logger.info(f"{C.grpc}[grpc]{C.r} RunTick RPC begin — draining client stream")
        try:
            bundle = self._assemble_bundle(request_iterator)
            logger.info(
                f"{C.grpc}[grpc]{C.r} bundle assembled tick=%d session=%s audio=%dB "
                "(sr=%d ch=%d) frames=%d ssl=%d heading=%.2frad",
                bundle.tick_id, bundle.session_id, len(bundle.audio_pcm),
                bundle.audio_sample_rate, bundle.audio_channels,
                len(bundle.frames), len(bundle.ssl_events),
                bundle.robot_heading_rad,
            )
        except Exception as e:
            logger.exception(f"{C.grpc}[grpc]{C.r} failed to assemble bundle: %s", e)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Failed to assemble tick bundle: {e}")
            return pb2.BraidTickResult()

        session = self._get_session(bundle.session_id)
        try:
            tick_res = run_tick(bundle, self.engine, self.gallery, session, self.cfg)
        except Exception as e:
            logger.exception(f"{C.grpc}[grpc]{C.r} run_tick failed: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"run_tick failed: {e}")
            return pb2.BraidTickResult(tick_id=bundle.tick_id,
                                       session_id=bundle.session_id)
        logger.info(f"{C.grpc}[grpc]{C.r} tick=%d persons=%d action=%s wall=%.2fs",
                    bundle.tick_id, len(tick_res.persons),
                    tick_res.action.type, time.time() - t0)
        return self._build_result(tick_res)
