from .api_base import ApiBase
from .speaking import _Speaking
from .silent import _Silent
from .person_attr import _PersonAttribute
from .bad_input import _BadInput
from .no_face import _NoFace
from .unsupported_action import _UnsupportedAction
from .secondary_channel import _SecondaryChannel
from .g1_gesture import _G1Gesture

Speaking = _Speaking()
Silent = _Silent()
PersonAttribute = _PersonAttribute()
BadInput = _BadInput()
NoFace = _NoFace()
UnsupportedAction = _UnsupportedAction()
SecondaryChannel = _SecondaryChannel()
G1Gesture = _G1Gesture()

api_call: dict[str, ApiBase] = {
    "speak": Speaking,
    "silent": Silent,
    "vision": PersonAttribute,
    "g1 unsupported action": UnsupportedAction,
    # Pepper's motion states, kept only as aliases. The G1 has none of
    # Pepper's joints, so a request that still reaches them must be
    # declined rather than answered with joint angles. They stay mapped
    # because the executor picks the closest key by fuzzy match: removing
    # them would route a motion request to whatever looked similar.
    "custom movement": UnsupportedAction,
    "standard movement": UnsupportedAction,
    "bad input": BadInput,
    "no face": NoFace,
    "object find": SecondaryChannel,
    "g1 wave": G1Gesture,
    "g1 handshake": G1Gesture,
    "g1 high five": G1Gesture,
    "g1 blow kiss left": G1Gesture,
    "g1 blow kiss right": G1Gesture,
    "g1 clap": G1Gesture,
    "g1 hug": G1Gesture,
    "g1 hand on heart": G1Gesture,
    "g1 waist drum dance": G1Gesture,
    "g1 spin discs": G1Gesture,
    "g1 throw money": G1Gesture,
    "g1 confirm wave": G1Gesture,
    "g1 confirm handshake": G1Gesture,
    "g1 confirm high five": G1Gesture,
}

__all__ = ["api_call"]
