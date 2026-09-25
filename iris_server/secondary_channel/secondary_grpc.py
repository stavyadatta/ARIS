import cv2
import json
import traceback
import numpy as np

from grpc_pb2 import TextChunk 
from grpc_pb2_grpc import SecondaryChannelServicer

from .secondary_channel import SecondaryChannel

class SecondaryGRPC(SecondaryChannelServicer):
    def __init__(self):
        super().__init__()

    def _decode_image_from_bytes(self, image_bytes):
        """
        Decode image bytes received in AudioImgRequest into a NumPy array usable by OpenCV.

        Args:
            image_bytes (bytes): The raw image data in bytes.

        Returns:
            np.ndarray: Decoded image as a NumPy array.
        """
        try:
            # Convert bytes to a 1D NumPy array
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            # Decode the image array into a format usable by OpenCV
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode the image from bytes.")
            return image
        except Exception as e:
            print("Error decoding image: {}".format(e))
            return None


    def Secondary_media_manager(self, request, context):
        try:
            api_task = json.loads(request.api_task)
            if request.HasField("image"):
                image_bytes = request.image.image_data
                print("the api task is ", api_task, "\n \n")
                image = self._decode_image_from_bytes(image_bytes)
                response = SecondaryChannel(img=image, api_task=api_task)
                mode = response.mode
                return TextChunk(
                    text=str(response.textchunk),
                    is_final=False,
                    mode=mode
                )
        except Exception as e:
            error_trace = traceback.format_exc()
            print("Error occurred while processing data: {}".format(error_trace))

            return TextChunk(
                text="error",
                is_final=True,
                mode='error'
            )

