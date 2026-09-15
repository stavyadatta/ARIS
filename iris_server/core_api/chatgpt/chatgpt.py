import openai

from ..vision_request import _VisionRequestMixin

# A person is standing in front of the robot waiting for an answer, so a
# stalled request must fail fast enough to fall back to another provider.
# The SDK defaults are a 600 s timeout and 2 retries, which turned one
# exhausted-quota response into a 4 s silence and could hold a gRPC worker
# for ten minutes.
REQUEST_TIMEOUT_SECONDS = 20
MAX_RETRIES = 1


class _OpenAIHandler(_VisionRequestMixin):
    vision_model = "gpt-4o"

    def __init__(self, model_name="gpt-4"):
        """
        Initialize the OpenAIHandler.

        :param model_name: The model name to use, e.g., "gpt-4"
        """
        self.client = openai.OpenAI(
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
        )


    def get_openai_embedding(self, text):
        """Generates OpenAI embedding for a given text."""
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding


    def send_o1(self, messages: list[dict], stream: bool, img=None, model="o1-preview"):
        """
            :param messages: A dictionary of messages for additional context to be 
             provided to the model for benefit
            :param stream: Whether to stream the output or not
            :param img: incase of VLM adding an image for additional context

            :return: Generator of words from llm incase of stream otherwise whole text 
                output
        """
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream
        )


    def send_text(self, messages: list[dict], stream: bool, img=None, model="gpt-4o", max_tokens=500):
        """
            :param messages: A dictionary of messages for additional context to be 
             provided to the model for benefit
            :param stream: Whether to stream the output or not
            :param img: incase of VLM adding an image for additional context

            :return: Generator of words from llm incase of stream otherwise whole text 
                output
        """
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=stream
        )

    def img_text_response(self, image, text, max_tokens=1000, system_prompt=None):
        """
        Process an image and text prompt using OpenAI API with streaming.

        :param image: NumPy array (from cv2), image path, or file-like object
        :param text: string with the user message
        :param max_tokens: Maximum tokens for response
        :returns : returns chunks

        """
        img_base64 = self._encode_image(image)
        img_text_dict = self.develop_last_message(text, img_base64)
        if system_prompt == None:
            system_prompt = self._develop_image_system_prompt()

        try:
            # Start streaming response
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    img_text_dict
                ],
                max_tokens=max_tokens,
                stream=False
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"Unexpected Error: {str(e)}"


    def send_text_get_json(self, messages: list[dict], stream: bool, img=None, max_tokens=500, model="gpt-4o"):
        """
            :param messages: A dictionary of messages for additional context to be 
             provided to the model for benefit
            :param stream: Whether to stream the output or not
            :param img: incase of VLM adding an image for additional context

            :return: Generator of words from llm incase of stream otherwise whole text 
                output
        """
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=stream,
            response_format={"type": "json_object"}
        )


