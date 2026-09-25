from utils import SecondaryDetails, ApiObject

class BaseSecondaryApi():
    def __init__(self):
        pass
    
    def __call__(self, secondary_details: SecondaryDetails) -> ApiObject:
        return ApiObject()
