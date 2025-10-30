import qi
import time
import json
import random

class StandardMovement:
    def __init__(self, session):
        self.animation_service = session.service("ALAnimationPlayer")
        print("Initialized ALAnimationPlayer")

    def extract_json_from_text(self, text):
        """
        Extracts and parses JSON embedded within a text string.
        
        :param text: The input string containing JSON and/or other text.
        :return: The parsed JSON as a Python dictionary if found.
        :raises ValueError: If no valid JSON is found in the text.
        """
        # Regular expression to find JSON-like structures in the text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print("Entering the error")
            raise ValueError("Found a JSON-like structure, but it is not valid JSON.")

    def choosing_random_dances(self):
        dance_types = [
            "asereje",
            "danceoftheknights",
            "danceofthereedflutes",
            "dango3kyodai",
            "Disco",
            "HeadBang",
            "hungarian",
            "jgangnamstyle",
            "la_bamba",
            "mika_relax",
            "walk_this_way"
        ]
        return random.choice(dance_types)


    def perform_dance(self, animation_name):
        self.animation_service.run(f"hapy_dance/{animation_name}")

    def perform_body_speech(self, animation_num):
        self.animation_service.run(f"animations/Stand/BodyTalk/Speaking/BodyTalk_{animation_num}")

    def __call__(self, llm_response):
        dance_type = self.choosing_random_dances()
        print("Now performing ", dance_type)
        self.perform_dance(dance_type)
