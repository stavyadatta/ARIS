from .objecting_looking import _ObjectLookup
from .base_secondary_api import BaseSecondaryApi

ObjectLookup = _ObjectLookup()

secondary_apis_kit: dict[str, BaseSecondaryApi] = {
    "ObjectLookup": ObjectLookup
}

__all__ = ["secondary_apis_kit"]

