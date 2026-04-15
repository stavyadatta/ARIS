"""Map ``BraidAction`` (gRPC) → Pepper ``ALMotion`` calls.

Rotations are executed via ``motion.moveTo(0, 0, ±θ)``; forward translation
via ``motion.moveTo(0.3, 0, 0)``; STAY is a no-op.
"""
from __future__ import annotations

import logging
import math
import time

logger = logging.getLogger("braid.client")


class ActionExecutor:
    def __init__(self, session):
        self.motion = session.service("ALMotion")

    def _current_heading(self) -> float:
        try:
            # (x, y, theta) in world frame
            pos = self.motion.getRobotPosition(False)
            return float(pos[2])
        except Exception:
            return 0.0

    def execute(self, action) -> float:
        """Execute one BraidAction protobuf. Returns the new robot heading
        (radians, best-effort world frame) after the motion completes."""
        atype = action.type  # int enum; compare by gRPC value via pb2
        mag = float(action.magnitude)
        try:
            # Avoid importing pb2 here to keep the client light; use string name.
            type_name = self._type_name(atype)
        except Exception:
            type_name = "STAY"

        logger.info("[action] executing %s magnitude=%.2f reason=%s target=%s",
                    type_name, mag, action.reason, action.target_person_id)

        try:
            self.motion.setStiffnesses("Body", 1.0)
            if type_name == "ROTATE_LEFT":
                self.motion.moveTo(0.0, 0.0, +mag)
            elif type_name == "ROTATE_RIGHT":
                self.motion.moveTo(0.0, 0.0, -mag)
            elif type_name == "MOVE_FORWARD":
                self.motion.moveTo(float(mag), 0.0, 0.0)
            elif type_name == "STAY":
                pass
            else:
                logger.warning("[action] unknown type %r; STAY", type_name)
        except Exception as e:
            logger.exception("[action] motion failed: %s", e)
        finally:
            try:
                self.motion.setStiffnesses("Body", 0.0)
            except Exception:
                pass
        # small settle
        time.sleep(0.2)
        return self._current_heading()

    @staticmethod
    def _type_name(value) -> str:
        # Accept either pb2 enum int or string name.
        if isinstance(value, str):
            return value
        from grpc_communication import grpc_pb2 as pb2
        return pb2.ActionType.Name(int(value))
