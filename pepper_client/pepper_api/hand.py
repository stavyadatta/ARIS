import qi
import time
import math

DEG_TO_RAD = math.pi / 180


class HandManager:
    """Controls Pepper's hand/arm movements for gestures like raising a hand."""

    def __init__(self, session):
        self.motion_service = session.service("ALMotion")
        self.posture_service = session.service("ALRobotPosture")
        print("Subscribed to hand service...")

    def raise_right_hand(self, speed=0.3):
        """Raises Pepper's right arm up as a hand-raise gesture.

        Params:
            speed: float (0 to 1)
                How fast the arm moves. Default 0.3 for a smooth motion.
        """
        joint_names = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw"]
        # ShoulderPitch: negative = arm goes up; -70 deg raises arm high
        # ShoulderRoll: slight inward so arm doesn't stick out awkwardly
        # ElbowYaw: rotated so hand faces forward
        # ElbowRoll: straighten the elbow
        # WristYaw: neutral
        # Go to init posture first so the arm starts from a known position
        self.posture_service.goToPosture("StandInit", speed)

        angles_deg = [-70.0, -15.0, 70.0, 2.0, 0.0]
        angles_rad = [a * DEG_TO_RAD for a in angles_deg]

        self.motion_service.angleInterpolationWithSpeed(joint_names, angles_rad, speed)
        time.sleep(2)

    def lower_right_hand(self, speed=0.3):
        """Returns the right arm to the default resting position."""
        self.posture_service.goToPosture("StandInit", speed)
