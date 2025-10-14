#!/usr/bin/env python3
import qi
import argparse
import sys
import time

def main(ip, port, distance, speed):
    connection_url = f"tcp://{ip}:{port}"
    app = qi.Application(["pepper_walk", f"--qi-url={connection_url}"])
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

    # Wake up & prepare
    motion.wakeUp()
    motion.setStiffnesses("Body", 1.0)
    # Allow arms to move for balance
    motion.setMoveArmsEnabled(True, True)
    # Stand straight
    posture.goToPosture("StandInit", 0.5)
    time.sleep(1.0)

    # Walk forward
    try:
        # speed is in m/s; Pepper's max is ~0.5 m/s
        config = [["MaxVelXY", speed]]
        motion.moveTo(distance, 0.0, 0.0, config)
        print(f"✅ Walked forward {distance:.2f} m at {speed:.2f} m/s")
    except RuntimeError as e:
        print("❌ moveTo failed:", e)

    # Sit and rest
    posture.goToPosture("Sit", 0.5)
    motion.rest()
    app.stop()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ip",       type=str,   default="192.168.0.52", help="Pepper IP")
    p.add_argument("--port",     type=int,   default=9559,          help="Naoqi port")
    p.add_argument("--distance", type=float, default=0.5,           help="Meters to walk")
    p.add_argument("--speed",    type=float, default=0.2,           help="Speed (m/s)")
    args = p.parse_args()
    main(args.ip, args.port, args.distance, args.speed)

