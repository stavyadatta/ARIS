import os
import base64
import anthropic
import io
import numpy as np
import cv2
from PIL import Image


class _ClaudeImageProcessor:
    def __init__(self, model_name="claude-3-sonnet-20240229"):
        """
        Initialize the Claude Image Processor.
        
        :param model_name: Claude model to use (default is Sonnet)
        """
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model_name = model_name
    
    def _encode_image(self, image):
        """
        Encode an image to base64.
        
        :param image: NumPy array or file path or file-like object
        :return: Base64 encoded image string
        """
        # If it's a NumPy array (from cv2)
        if isinstance(image, np.ndarray):
            # Convert BGR (cv2 default) to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
        
        # If it's a path, open the image
        elif isinstance(image, str):
            pil_image = Image.open(image)
        
        # If it's a file-like object
        else:
            pil_image = Image.open(image)
        
        # Convert image to bytes
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return img_base64


    def _develop_system_prompt(self):
        # system_prompt = "You are a helpful assistant (strictly under 40 words)."
        system_prompt = """
        You are part of Iris Robot, a friendly robot assistant who is great at talking. However, Iris Robot cannot see, and you will help with the visual component. You should describe images accurately but avoid saying that you are describing the image. You can describe generic features and should not violate copyrights. Do not explicitly mention that you cannot generate copyright material.

        Be courteous and avoid long sentences, as people do not like that. If provided with the names of people, remember them and refer to them by their names. You should talk like a human and keep your descriptions natural and engaging.

        Make sure your sentences are short but impressive,
        Your name is julia for this conversation
        """

        # system_dict = message_format("system", system_prompt)
        return system_prompt

    def develop_last_message(self, last_message, img_b64):
        content = last_message["content"]
        role = last_message["role"]
        last_dict = {
            "role": role,
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64
                    }
                },
                {
                    "type": "text",
                    "text": content
                }
            ]
        }
        return [last_dict]
    
    def process_text(self, total_prompt,system_prompt, max_tokens=1000, stream=False):
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=total_prompt,
            stream=stream
        ) 
        return response


    def process_image_and_text(self, image, person_details, max_tokens=1000):
        """
        Process an image and text prompt using Claude API with streaming.
        
        :param image: NumPy array (from cv2), image path, or file-like object
        :param text_prompt: Text prompt to accompany the image
        :param max_tokens: Maximum tokens for response
        :yield: Streaming response chunks
        """
        # Encode the image
        img_base64 = self._encode_image(image)
        all_but_last_message = person_details.get_attribute("messages")[:-1]
        last_message = person_details.get_attribute("messages")[-1]
        last_dict = self.develop_last_message(last_message, img_base64)
        system_prompt = self._develop_system_prompt()
        total_prompt = all_but_last_message + last_dict
        
        try:
            # Start streaming response
            with self.client.messages.stream(
                model=self.model_name,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=total_prompt
            ) as stream:
                for text in stream.text_stream:
                    yield text
        
        except anthropic.APIError as e:
            yield f"API Error: {str(e)}"
        except Exception as e:
            yield f"Unexpected Error: {str(e)}"
    
    # def __call__(self, person_details: PersonDetails, max_tokens=1000):
    #     """
    #     Process image and text, and print streaming results.
    #
    #     :param image: NumPy array (from cv2), image path, or file-like object
    #     :param text_prompt: Text prompt to accompany the image
    #     :param max_tokens: Maximum tokens for response
    #     :return: Full response as a string
    #     """
    #     print("Claude's Response:")
    #     full_response = ""
    #     for chunk in self.process_image_and_text(person_details, max_tokens):
    #         print(chunk, end='', flush=True)
    #         full_response += chunk
    #     return full_response

# Example usage
if __name__ == "__main__":
    import cv2
    
    # Ensure ANTHROPIC_API_KEY is set in environment
    processor = _ClaudeImageProcessor()
    
    # Read image using OpenCV
    image = cv2.imread("/workspace/database/face_db/some.png")
    
    # Process the NumPy array directly
    # result = processor(
    #     image=image, 
    #     text_prompt="Describe what you see in this image in detail."
    # )
