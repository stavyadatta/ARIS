"""BRAID client — 30s capture → RunTick → execute action → repeat.

Invocation (on the robot's host machine, NOT this dev env):

    python -m pepper_client.braid.braid_client \
        --braid --server 192.168.1.100:50051 \
        --robot-ip 192.168.0.52 --robot-port 9559 \
        --session-id demo_session --num-ticks 3

The ``--braid`` flag is accepted both as an explicit on-switch (for symmetry
with the server and to make the activation pattern obvious at the call site)
and as an implicit default: if this module is invoked at all, BRAID mode is on.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
import uuid
from typing import Iterator

import grpc
import qi  # type: ignore

# Repo layout: we import the generated protobuf from the /workspace
# grpc_communication package.
from grpc_communication import grpc_pb2 as pb2
from grpc_communication import grpc_pb2_grpc as pb2_grpc

from pepper_client.braid.action_executor import ActionExecutor
from pepper_client.braid.bundle_capture import BundleCapture, ClientBundle


logger = logging.getLogger("braid.client")


# -------- chunker: stream the ClientBundle to the server ----------------------

AUDIO_CHUNK_BYTES = 64 * 1024  # ~2s at 16kHz mono int16; safe under default gRPC max msg


def _chunk_iter(bundle: ClientBundle) -> Iterator[pb2.BraidTickChunk]:
    # 1) meta
    meta = pb2.BraidMeta(
        session_id=bundle.session_id,
        tick_id=bundle.tick_id,
        window_start_ts=float(bundle.window_start_ts),
        robot_heading_rad=float(bundle.robot_heading_rad),
        audio_sample_rate=int(bundle.audio_sample_rate),
        audio_num_channels=int(bundle.audio_channels),
    )
    yield pb2.BraidTickChunk(
        tick_id=bundle.tick_id,
        session_id=bundle.session_id,
        robot_heading_rad=float(bundle.robot_heading_rad),
        meta=meta,
    )

    # 2) audio
    buf = bundle.audio_pcm
    for i in range(0, len(buf), AUDIO_CHUNK_BYTES):
        piece = buf[i:i + AUDIO_CHUNK_BYTES]
        yield pb2.BraidTickChunk(
            tick_id=bundle.tick_id,
            session_id=bundle.session_id,
            robot_heading_rad=float(bundle.robot_heading_rad),
            audio_chunk=pb2.BraidAudioChunk(
                pcm=piece,
                ts=float(bundle.window_start_ts + i / (2.0 * bundle.audio_sample_rate
                                                        * max(1, bundle.audio_channels))),
                channels=int(bundle.audio_channels),
                sample_rate=int(bundle.audio_sample_rate),
            ),
        )

    # 3) frames
    for ts, jpg, w, h in bundle.frames:
        yield pb2.BraidTickChunk(
            tick_id=bundle.tick_id,
            session_id=bundle.session_id,
            robot_heading_rad=float(bundle.robot_heading_rad),
            frame_chunk=pb2.BraidFrameChunk(
                jpeg=jpg, ts=float(ts), width=int(w), height=int(h),
            ),
        )

    # 4) ssl events
    for ts, az, el, conf in bundle.ssl_events:
        yield pb2.BraidTickChunk(
            tick_id=bundle.tick_id,
            session_id=bundle.session_id,
            robot_heading_rad=float(bundle.robot_heading_rad),
            ssl_event=pb2.SslEvent(
                azimuth_rad=float(az), elevation_rad=float(el),
                confidence=float(conf), ts=float(ts),
            ),
        )


# -------- main loop ----------------------------------------------------------

class BraidClient:
    def __init__(self, session, server: str, session_id: str,
                 tick_seconds: float = 30.0):
        self.session = session
        self.channel = grpc.insecure_channel(server)
        self.stub = pb2_grpc.BraidServiceStub(self.channel)
        self.capture = BundleCapture(session)
        self.action = ActionExecutor(session)
        self.session_id = session_id
        self.tick_seconds = tick_seconds
        self.heading = self.action._current_heading()

    def run_once(self, tick_id: int):
        logger.info("[client] ===== tick %d (session=%s heading=%.2frad) =====",
                    tick_id, self.session_id, self.heading)
        bundle = self.capture.run(
            tick_id=tick_id, session_id=self.session_id,
            duration_seconds=self.tick_seconds,
            robot_heading_rad=self.heading,
        )
        t0 = time.time()
        result = self.stub.RunTick(_chunk_iter(bundle))
        logger.info("[client] server returned %d persons, action=%s, wall=%.2fs",
                    len(result.persons), pb2.ActionType.Name(result.action.type),
                    time.time() - t0)
        for pd in result.persons:
            logger.info(
                "  person=%s state=%s p_best=%.2f p_unk=%.2f margin=%.2f "
                "entropy=%.2f identity=%s mod_agree=%s face_quality=%.2f reason=%s",
                pd.person_id, pb2.DecisionState.Name(pd.state),
                pd.p_best, pd.p_unk, pd.margin, pd.entropy,
                pd.identity or "-", pd.modality_agreement, pd.face_quality,
                pd.reason,
            )
        # Execute action.
        new_heading = self.action.execute(result.action)
        self.heading = new_heading
        return result

    def run_forever(self, num_ticks: int = 0):
        i = 0
        while True:
            i += 1
            try:
                self.run_once(i)
            except KeyboardInterrupt:
                logger.info("[client] interrupted")
                break
            except Exception as e:
                logger.exception("[client] tick %d failed: %s", i, e)
                time.sleep(1.0)
            if num_ticks and i >= num_ticks:
                logger.info("[client] reached num_ticks=%d, exiting", num_ticks)
                break


# -------- entry point --------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="BRAID pepper client")
    p.add_argument("--braid", action="store_true",
                   help="Explicit on-switch (redundant; this script implies BRAID).")
    p.add_argument("--server", type=str, default="localhost:50051",
                   help="ginny_server gRPC endpoint")
    p.add_argument("--robot-ip", type=str, default="192.168.0.52")
    p.add_argument("--robot-port", type=int, default=9559)
    p.add_argument("--session-id", type=str, default=None,
                   help="Optional session id (uuid4 by default).")
    p.add_argument("--tick-seconds", type=float, default=30.0)
    p.add_argument("--num-ticks", type=int, default=0,
                   help="0 = run forever.")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    session_id = args.session_id or f"braid_{uuid.uuid4().hex[:8]}"

    url = f"tcp://{args.robot_ip}:{args.robot_port}"
    app = qi.Application(["BraidClient", f"--qi-url={url}"])
    app.start()
    session = app.session

    client = BraidClient(session=session, server=args.server,
                         session_id=session_id,
                         tick_seconds=float(args.tick_seconds))
    logger.info("BRAID client connected to server=%s robot=%s session=%s",
                args.server, url, session_id)
    client.run_forever(num_ticks=args.num_ticks)


if __name__ == "__main__":
    main()
