from core_api import YOLODetector
from utils import SecondaryDetails, ApiObject

from .base_secondary_api import BaseSecondaryApi

class _ObjectLookup(BaseSecondaryApi):
    def __init__(self):
        pass
    
    def __call__(self, secondary_details: SecondaryDetails) -> ApiObject:
        img = secondary_details.img
        api_detail = secondary_details.api_detail
        print("The api detail is ", api_detail)
        object_name = api_detail.get("object_name")
        assert object_name is not None

        YOLODetector.set_classes([object_name])
        output = YOLODetector.detect_objects(img)
        if len(output) == 0:
            empty_bbox = {'bbox': ""}
            return ApiObject(textchunk=empty_bbox, mode="not done")
        
        bbox = output[0].get('bounding_box')
        bbox_dict = {'bbox': bbox}
        bbox_str = str(bbox_dict)
        api_object = ApiObject(textchunk=bbox_str, mode='done')
        return api_object
