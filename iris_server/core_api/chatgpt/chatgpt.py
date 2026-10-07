import os

import openai

from ..vision_request import _VisionRequestMixin

# A person is standing in front of the robot waiting for an answer, so a
# stalled request must fail fast enough to fall back to another provider.
# The SDK defaults are a 600 s timeout and 2 retries, which turned one
# exhausted-quota response into a 4 s silence and could hold a gRPC worker
# for ten minutes.
REQUEST_TIMEOUT_SECONDS = 20
MAX_RETRIES = 1

# The model behind every chat reply and classification. Set OPENAI_MODEL in
# .env to change it without touching code; a smaller model answers faster.
# Vision and o1 requests keep their own models.
DEFAULT_CHAT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# The model that turns a spoken request for several gestures into an ordered
# queue. Set IRIS_PLANNER_MODEL to use a different (smaller) one; unset or
# empty, it follows OPENAI_MODEL. It must support strict JSON schema output.
DEFAULT_PLANNER_MODEL = os.environ.get("IRIS_PLANNER_MODEL") or DEFAULT_CHAT_MODEL


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


    def send_text(self, messages: list[dict], stream: bool, img=None, model=DEFAULT_CHAT_MODEL, max_tokens=500):
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

    def send_structured(self, messages: list[dict], response_format: dict,
                        max_tokens: int, timeout: float,
                        model=DEFAULT_PLANNER_MODEL) -> str:
        """One deterministic request whose reply must follow a JSON schema.

        No retry: the caller has a fallback, and a retry would only make the
        person wait longer for it. Returns the reply's text.

        :param response_format: an OpenAI `json_schema` response format
        :param timeout: seconds before the request is abandoned
        """
        response = self.client.with_options(timeout=timeout, max_retries=0).chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
            response_format=response_format,
        )
        return response.choices[0].message.content

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


    def send_text_get_json(self, messages: list[dict], stream: bool, img=None, max_tokens=500, model=DEFAULT_CHAT_MODEL):
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


