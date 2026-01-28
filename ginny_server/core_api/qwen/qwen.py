import os
import openai
import base64
import cv2
import numpy as np

from utils import Neo4j

class _QwenHandler:
    def __init__(self, model_name="qwen"):
        """
        Initialize the QwenHandler.

        :param model_name: The model name to use, e.g., "qwen"
        """
        self.client = openai.OpenAI(
            api_key=os.getenv("QWEN_API_KEY", "EMPTY"),
            base_url=os.getenv("QWEN_API_BASE", "http://localhost:30000/v1"),
        )

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


    def send_text(self, messages: list[dict], stream: bool, img=None, qwen_model="qwen"):
        """
            :param messages: A dictionary of messages for additional context to be 
             provided to the model for benefit
            :param stream: Whether to stream the output or not
            :param img: incase of VLM adding an image for additional context

            :return: Generator of words from llm incase of stream otherwise whole text 
                output
        """
        # If the user passes a specific model, use it, otherwise use a default if needed.
        # However, SGLang usually serves one model, so the model name might be ignored or checked.
        # We'll pass what's given.
        
        response = self.client.chat.completions.create(
            model= qwen_model,
            messages= messages,
            temperature=0.7,
            max_tokens=500,
            top_p=0.9,
            stream=stream
        )
        if isinstance(response, str):
            raise Exception("Qwen did not respond, returned str ", response)
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
                model="qwen-vl", # Assumption for VL model
                messages=[
                    {"role": "system", "content": system_prompt},
                    img_text_dict
                ],
                max_tokens=max_tokens,
                stream=False
            )
            if isinstance(response, str):
                raise Exception("Qwen did not respond, returned str ", response)
            content = response.choices[0].message.content
            return content

        except openai.OpenAIError as e:
            return f"API Error: {str(e)}"
        except Exception as e:
            return f"Unexpected Error: {str(e)}"

    def process_image_and_text(self, image, person_details, max_tokens=1000, system_prompt=None):
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

        # Combine messages for the API call
        try:
            # Start streaming response
            response = self.client.chat.completions.create(
                model="qwen-vl",
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
        robot_description = """ 
          You are part of Ginny Robot, a friendly robot assistant who excels at talking. However, Ginny Robot does not have vision,
          and you assist with the visual component. GINNY robot may also be playing a game of 
          pictionary where you need to guess what the person sitting is describing, if the 
          person asks you to guess the item or place, or animal, you need to focus on the hint 
          they are provided and answer accordingly 

          When you receive the prompt from user you need to think in the following way

          1) Does the image have anything do with the prompt that the user has 
          sent 
          2) If yes then I should only reply with a description that was specifically 
          asked by the user,

          for example: 

          input: How do I look 
          output: In that black tshirt you look amazing

          input: How does these glasses look on me
          output: The round shaped glasses are loking great with your clean 
          beard

          3) if the prompt has nothing to do with the image then donot take 
          image into consideration

          for example:
          input: What do you think paris looks like 
          output: Paris looks pretty 

          input: What should I wear?
          output: I think you should wear something nice like black

          input: 

          4) I should sound human, similar to how people chat on facebook
          5) I should be concise with my responses
          6) When referring to an image or photo, replace those words with phrases like 'I see...'."
          7) Make sure in your response you are not giving justification of your reasoning
          8) Donot use, words like "image", "picture" or any of the synonyms
        """
        return robot_description

Qwen = _QwenHandler()
