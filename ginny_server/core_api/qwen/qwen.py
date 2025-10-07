from _typeshed import ExcInfo
import openai
import numpy as np

BASE_URL = "http://127.0.0.1:30000/v1"  # your sglang server
API_KEY = "EMPTY"

class _QwenHandler():
    def __init__(self, model_name="qwen3-1.7b") -> None:
        """ 
            Initialise the OpenAI handler for qwen

            :param model_name: The model name is qwen3-1-1.7b
        """

        self.client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)


    def send_text(self, messages: list[dict], stream: bool, img=None, qwen_model="qwen3-1"):
        """
            :param messages: A dictionary of messages for additional context to be 
             provided to the model for benefit
            :param stream: Whether to stream the output or not
            :param img: incase of VLM adding an image for additional context

            :return: Generator of words from llm incase of stream otherwise whole text 
                output
        """
        response = self.client.chat.completions.create(
            model=qwen_model,
            messages=messages,
            max_tokens=500,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                # When thinking is enabled, ask server to separate it.
                # If disabled, most servers just omit reasoning_content.
                "separate_reasoning": False,
            },
            stream=stream
        )

        if isinstance(response, str):
            raise Exception("Qwen server did not respond, returned error")
        return response
