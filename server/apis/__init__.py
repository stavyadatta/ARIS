from .api_base import ApiBase
from .speaking import _Speaking
from .silent import _Silent
from .person_attr import _PersonAttribute
from .bad_input import _BadInput
from .movement import _CustomMovement, _StandardMovement

Speaking = _Speaking()
Silent = _Silent()
PersonAttribute = _PersonAttribute()
BadInput = _BadInput()
CustomMovement = _CustomMovement()
StandardMovement = _StandardMovement()

api_call: dict[str, ApiBase] = {
    "speak": Speaking,
    "silent": Silent,
    "vision": PersonAttribute,
    "custom movement": CustomMovement,
    "standard movement": StandardMovement,
    "bad input": BadInput,
}

__all__ = ["api_call"]
