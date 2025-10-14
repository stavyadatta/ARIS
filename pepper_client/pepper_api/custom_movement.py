#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import qi
import sys
import time
import json
import math
import argparse

DEG_TO_RAD = math.pi / 180.0
SAFE_MAX_VEL_XY = 0.25   # m/s
SAFE_MIN_VEL_XY = 0.02   # m/s
SAFE_MAX_VEL_TH = 0.5    # rad/s

JOINT_HANDS = {"LHand", "RHand"}

def _is_hand(jname: str) -> bool:
    return jname in JOINT_HANDS

def _to_joint_target(jname: str, angle_deg: float) -> float:
    """Hands use 0..1; others are radians converted from degrees."""
    if _is_hand(jname):
        # Accept 0..1; if someone sent 0..100, normalise.
        if angle_deg > 1.0:
            return max(0.0, min(1.0, angle_deg / 100.0))
        return max(0.0, min(1.0, angle_deg))
    return float(angle_deg) * DEG_TO_RAD

class CustomMovement:
    def __init__(self, session, robot_posture):
        self.motion_service = session.service("ALMotion")
        self.posture_service = robot_posture
        print("Initialized ArmManager and woke up the robot.")

    def extract_json_from_text(self, text):
        """
        Extracts and parses JSON embedded within a text string.
        Returns a Python dict (must contain 'action_list').
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError("Found a JSON-like structure, but it is not valid JSON.")

    def movement(self, joint_names, joint_angles, speed):
        """
        Drives joints only (kept for backward compatibility).
        """
        for name, angle, sp in zip(joint_names, joint_angles, speed):
            target = _to_joint_target(name, angle)
            try:
                self.motion_service.angleInterpolationWithSpeed(name, target, float(sp))
            except Exception as e:
                print("An error occurred while moving the arm {}".format(e))

    # ---------- Private helpers (new) ----------
    def _do_posture(self, name: str, speed: float):
        try:
            self.posture_service.goToPosture(str(name), float(speed))
        except Exception as e:
            print(f"[posture] error: {e}")

    def _do_joint(self, joint_name: str, angle_deg: float, speed: float):
        try:
            target = _to_joint_target(joint_name, angle_deg)
            self.motion_service.angleInterpolationWithSpeed(str(joint_name), target, float(speed))
        except Exception as e:
            print(f"[joint] {joint_name} error: {e}")

    def _do_wait(self, seconds: float):
        try:
            time.sleep(max(0.0, float(seconds)))
        except Exception as e:
            print(f"[wait] error: {e}")

    def _do_locomotion(self, x: float, y: float, theta_deg: float, speed_mps: float):
        """
        Relative motion in Pepper's body frame.
        x: +forward/-back (m), y: +left/-right (m), theta_deg: +left/-right (deg).
        """
        try:
            vxy = max(SAFE_MIN_VEL_XY, min(SAFE_MAX_VEL_XY, float(speed_mps)))
            vth = min(SAFE_MAX_VEL_TH, max(0.2, vxy * 3.0))
            theta = float(theta_deg) * DEG_TO_RAD
            cfg = [["MaxVelXY", vxy], ["MaxVelTheta", vth]]
            self.motion_service.moveTo(float(x), float(y), theta, cfg)
        except Exception as e:
            print(f"[locomotion] error: {e}")

    # ---------- Main entrypoint you asked to keep ----------
    def __call__(self, llm_response):
        """
        Accepts an LLM JSON string and executes both stationary and locomotion steps.
        Supports two schemas:
          A) New (recommended):
             {"action_list":[
                {"type":"posture","name":"StandInit","speed":0.6,"reasoning":"..."},
                {"type":"joint","joint_name":"RHand","angle_deg":1.0,"speed":0.5,"reasoning":"..."},
                {"type":"locomotion","x":0.3,"y":0.0,"theta_deg":0,"speed":0.15,"reasoning":"..."},
                {"type":"wait","seconds":0.5,"reasoning":"..."}
             ]}
          B) Legacy (back-compat): a flat list of {"joint_name","angle","speed"} only.
        """
        action_json = self.extract_json_from_text(llm_response)
        print("The json action is \n \n \n \n ", action_json)

        action_items = action_json.get("action_list")
        if not isinstance(action_items, list):
            raise ValueError("JSON must contain 'action_list' as a list.")

        # Detect legacy schema (no 'type' keys → treat as joint batch)
        legacy = all(isinstance(a, dict) and "type" not in a for a in action_items)

        if legacy:
            joint_names, angles, speeds = [], [], []
            for action in action_items:
                joint_names.append(action.get("joint_name"))
                angles.append(action.get("angle"))
                speeds.append(action.get("speed"))
            self.movement(joint_names, angles, speeds)
            # Return to neutral after legacy batch
            try:
                self.posture_service.goToPosture("StandInit", 0.2)
            except Exception:
                pass
            time.sleep(2.0)
            return

        # New schema: step-by-step execution
        for idx, a in enumerate(action_items, 1):
            atype = a.get("type")
            try:
                if atype == "posture":
                    self._do_posture(a.get("name", "StandInit"), a.get("speed", 0.6))
                elif atype == "joint":
                    self._do_joint(a["joint_name"], a["angle_deg"], a.get("speed", 0.5))
                elif atype == "wait":
                    self._do_wait(a.get("seconds", 0.5))
                elif atype == "locomotion":
                    self._do_locomotion(
                        a.get("x", 0.0),
                        a.get("y", 0.0),
                        a.get("theta_deg", 0.0),
                        a.get("speed", 0.15),
                    )
                else:
                    print(f"[warn] step {idx}: unknown type '{atype}', skipping.")
            except Exception as e:
                print(f"[step {idx}] execution error: {e}")

        # Return to a neutral stable pose after plan
        try:
            self.posture_service.goToPosture("StandInit", 0.4)
        except Exception:
            pass
        time.sleep(1.0)


# ---------- Optional CLI harness (kept minimal; your larger program can ignore) ----------
def _read_json(stdin_fallback: bool, json_file: str = None) -> str:
    if json_file:
        with open(json_file, "r") as f:
            return f.read()
    if stdin_fallback:
        return sys.stdin.read()
    raise ValueError("Provide --json-file or pipe JSON on stdin.")

def main():
    parser = argparse.ArgumentParser(description="Execute Pepper movement plan JSON.")
    parser.add_argument("--ip", type=str, default="192.168.0.52", help="Robot IP address")
    parser.add_argument("--port", type=int, default=9559, help="Robot port number")
    parser.add_argument("--json-file", type=str, default=None, help="Path to JSON plan; if omitted, read from stdin.")
    args = parser.parse_args()

    try:
        session = qi.Session()
        session.connect("tcp://{}:{}".format(args.ip, args.port))
        print("Connected to Pepper robot.")
    except Exception as e:
        print("Cannot connect to Pepper robot at {}:{}. Error: {}".format(args.ip, args.port, e))
        sys.exit(1)

    try:
        robot_posture = session.service("ALRobotPosture")
    except Exception as e:
        print("Could not access ALRobotPosture: {}".format(e))
        sys.exit(1)

    cm = CustomMovement(session, robot_posture)

    try:
        llm_response_text = _read_json(stdin_fallback=(args.json_file is None), json_file=args.json_file)
        cm(llm_response_text)
    except Exception as e:
        print("Execution error: {}".format(e))
        sys.exit(1)

if __name__ == "__main__":
    main()

