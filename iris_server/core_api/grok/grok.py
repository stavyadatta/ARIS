import os

from ..chatgpt.chatgpt import MAX_RETRIES, REQUEST_TIMEOUT_SECONDS

import openai

from ..vision_request import _VisionRequestMixin

class _GrokHandler(_VisionRequestMixin):
    vision_model = "grok-2-vision-1212"

    def __init__(self, model_name="gpt-4"):
        """
        Initialize the OpenAIHandler.

        :param model_name: The model name to use, e.g., "gpt-4"
        """
        self.client = openai.OpenAI(
            api_key=os.getenv("GROK_API_KEY"),
            base_url="https://api.x.ai/v1",
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
        )


    def send_text(self, messages: list[dict], stream: bool, img=None, grok_model="grok-2"):
        """
            :param messages: A dictionary of messages for additional context to be 
             provided to the model for benefit
            :param stream: Whether to stream the output or not
            :param img: incase of VLM adding an image for additional context

            :return: Generator of words from llm incase of stream otherwise whole text 
                output
        """
        response = self.client.chat.completions.create(
            model= grok_model,
            messages= messages,
            temperature=0.7,
            max_tokens=500,
            top_p=0.9,
            stream=stream
        )
        if isinstance(response, str):
            raise Exception("Grok did not respond, returned str ", response)
        return response

    def img_text_response(self, image, text, max_tokens=1000, system_prompt=None):
        """
        Process an image and text prompt using OpenAI API with streaming.

        :param image: NumPy array (from cv2), image path, or file-like object
        :param text: string with the user message
        :param max_tokens: Maximum tokens for response
        :returns: returns response 
        """
        print("Is it entring img_text_response\n\n")
        img_base64 = self._encode_image(image)
        img_text_dict = self.develop_last_message(text, img_base64)
        if system_prompt == None:
            system_prompt = self._develop_image_system_prompt()

        try:
            # Start streaming response
            response = self.client.chat.completions.create(
                model="grok-2-vision-1212",
                messages=[
                    {"role": "system", "content": system_prompt},
                    img_text_dict
                ],
                max_tokens=max_tokens,
                stream=False
            )
            if isinstance(response, str):
                raise Exception("chatgpt did not respond, returned str ", response)
            content = response.choices[0].message.content
            print("The content is ", content)
            return content

        except Exception as e:
            return f"Unexpected Error: {str(e)}"


