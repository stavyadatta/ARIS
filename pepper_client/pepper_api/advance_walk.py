#!/usr/bin/env python3
import qi
import argparse
import sys
import time

def main(ip, port, distance, speed, direction):
    connection_url = f"tcp://{ip}:{port}"
    app = qi.Application(["pepper_move", f"--qi-url={connection_url}"])
    try:
        app.start()
    except RuntimeError:
        print(f"❌ Cannot connect to Pepper at {ip}:{port}")
        sys.exit(1)

    session = app.session
    motion = session.service("ALMotion")
    posture = session.service("ALRobotPosture")
    life = session.service("ALAutonomousLife")

    # Disable autonomous behaviors so we’re in direct control
    life.setState("disabled")

    # Wake up and prepare
    motion.wakeUp()
    motion.setStiffnesses("Body", 1.0)
    motion.setMoveArmsEnabled(True, True)
    posture.goToPosture("StandInit", 0.5)
    time.sleep(1.0)

    # Determine movement vector
    x, y, theta = 0.0, 0.0, 0.0
    if direction == "forward":
        x = distance
    elif direction == "backward":
        x = -distance
    elif direction == "left":
        y = distance
    elif direction == "right":
        y = -distance
    else:
        print("❌ Invalid direction. Use forward/backward/left/right.")
        app.stop()
        return

    try:
        config = [["MaxVelXY", speed]]
        motion.moveTo(x, y, theta, config)
        print(f"✅ Moved {direction} by {distance:.2f} m at {speed:.2f} m/s")
    except RuntimeError as e:
        print("❌ moveTo failed:", e)

    # Sit and rest
    posture.goToPosture("Sit", 0.5)
    motion.rest()
    app.stop()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ip", type=str, default="192.168.0.52", help="Pepper IP")
    p.add_argument("--port", type=int, default=9559, help="Naoqi port")
    p.add_argument("--distance", type=float, default=0.3, help="Meters to move")
    p.add_argument("--speed", type=float, default=0.15, help="Speed (m/s)")
    p.add_argument("--direction", type=str, choices=["forward", "backward", "left", "right"], default="forward",
                   help="Movement direction")
    args = p.parse_args()
    main(args.ip, args.port, args.distance, args.speed, args.direction)

