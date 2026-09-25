"""Vision plumbing shared by the OpenAI-compatible handlers.

Grok speaks the OpenAI wire format, so encoding the frame, shaping the
image message and streaming the answer back are one behaviour for both
providers; only the client and the model id differ. Keeping them together is
what lets the vision path fail over from one provider to the other without
the robot describing the scene differently.

Subclasses supply `self.client` and a `vision_model`.
"""

import base64

import cv2
import numpy as np
import openai

from utils import Neo4j

from .vision_prompt import VISION_SYSTEM_PROMPT


class _VisionRequestMixin:
    #: Model this handler asks to look at frames. Subclasses must set it.
    vision_model: str = ""

    def _encode_image(self, image):
        """
        Encode an image into base64 format for processing.

        :param image: NumPy array (from cv2), image path, or file-like object
        :return: Base64 encoded image string
        """
        if isinstance(image, np.ndarray):
            _, buffer = cv2.imencode(".png", image)
            encoded_image = base64.b64encode(buffer).decode("utf-8")
        elif isinstance(image, str):
            with open(image, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
        else:
            encoded_image = base64.b64encode(image.read()).decode("utf-8")

        return encoded_image

    def process_image_and_text(self, image, person_details, max_tokens=1000,
                               system_prompt=None, model_name=None):
        """
        Process an image and text prompt using OpenAI API with streaming.

        :param image: NumPy array (from cv2), image path, or file-like object
        :param person_details: An object with a `get_attribute` method for accessing messages
        :param max_tokens: Maximum tokens for response
        :yield: Streaming response chunks
        """
        # Encode the image
        face_id = person_details.get_attribute("face_id")
        img_base64 = self._encode_image(image)
        last_message = person_details.get_latest_user_message()
        messages = Neo4j.get_person_messages(last_message, face_id)

        # Develop the last message including the image
        last_dict = self.develop_last_message(last_message, img_base64)

        # Create the system prompt
        if system_prompt == None:
            system_prompt = self._develop_image_system_prompt()

        if model_name is None:
            model_name = self.vision_model

        # Combine messages for the API call
        try:
            # Start streaming response
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt}] + messages + [last_dict],
                max_tokens=max_tokens,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except openai.OpenAIError as e:
            yield f"API Error: {str(e)}"
        except Exception as e:
            yield f"Unexpected Error: {str(e)}"

    def develop_last_message(self, last_message, img_base64):
        """
        Create the last message dictionary with the image included.

        :param last_message: Last message details
        :param img_base64: Base64 encoded image string
        :return: Updated last message dictionary
        """
        if isinstance(last_message, str):
            return {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": last_message
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    }
                ]
            }
        else:
            return {
                "role": last_message.get("role"),
                "content": [
                    {
                        "type": "text",
                        "text": last_message.get("content")
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    }
                ]
            }

    def _develop_image_system_prompt(self):
        """
        Generate the system-level prompt.

        :return: System prompt string
        """
        return VISION_SYSTEM_PROMPT
