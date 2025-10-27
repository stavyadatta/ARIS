import re
import sys
import json
import time
import random
import threading
from google.protobuf.empty_pb2 import Empty

import grpc
from collections import deque

from secondary_communication import SecondaryCommunication

def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except Exception:
        return False

class SpeechProcessor:
    def __init__(self, speech_function, standard_movement):
        self.sentence_queue = deque()  # Queue for sentences
        self.current_sentence = ""
        self.speech_function = speech_function  # Instance of Pepper's speech manager
        self.is_running = True  # Flag to control the threads
        self.to_execute_movement_thread = True
        self.movement = False
        self.standard_movement = standard_movement
        self.do_movement = threading.Event()
        self.body_thread = threading.Thread(target=self.body_movement)
        self.body_thread.start()

    def body_movement(self):
    # This loop will keep the thread alive as long as self.is_running is True.
        movement_num = 1
        while self.to_execute_movement_thread:
            if self.do_movement.is_set():
                # If the event is set, perform a movement
                self.standard_movement.perform_body_speech(movement_num)
                movement_num += 1
                if movement_num > 16:
                    movement_num = 1
            else:
                # When not moving, you might want to sleep briefly to prevent busy-waiting
                time.sleep(0.1)

    def build_sentences(self, response_stream):
        """
        Build sentences from streamed words and add them to the queue.
        :param response_stream: gRPC response stream from LLM.
        """
        try:
            for chunk in response_stream:
                sys.stdout.write(chunk.text + "")
                sys.stdout.flush()

                # if chunk.is_final == True:
                #     break

                # Append chunk words to the current sentence
                self.current_sentence += chunk.text

                # Check for sentence-ending punctuation
                if is_valid_json(self.current_sentence):
                    self.sentence_queue.append((self.current_sentence, chunk.mode))
                elif re.search(r'[.!?]$', chunk.text.strip()):
                    # Join the words to form a complete sentence
                    self.sentence_queue.append((self.current_sentence.strip(), chunk.mode))  # Add to queue
                    self.current_sentence = ""

                # Exit if the flag is turned off
                if not self.is_running:
                    break
        except grpc.RpcError as e:
            print("gRPC LLM response error: {} - {}".format(e.code(), e.details()))

    def execute_response(self, pepper):
        """
        Continuously retrieve sentences from the queue and make Pepper speak them.
        """
        print("execute response first \n \n \n")
        print(f"Status of flags is running {self.is_running}, sentence_queue {bool(self.sentence_queue)} \n")
        while self.is_running or self.sentence_queue:
            if self.sentence_queue:
                sentence_tuple = self.sentence_queue.popleft()
                sentence_to_say = sentence_tuple[0]
                mode = sentence_tuple[1]
                print("Coming inside this part")
                if not is_valid_json(sentence_to_say):
                    print("Coming inside the not valid json part?")
                    self.do_movement.set()
                    self.speech_function(sentence_to_say)
                else:
                    if mode == "secondary":
                        SecondaryCommunication(sentence_to_say, mode, pepper)
                    elif mode == "custom_movement":
                        pepper.custom_movement(sentence_to_say)
                    elif mode == "standard_movement":
                        pepper.standard_movement(sentence_to_say)
                    elif mode == "pepper_auto":
                        pepper.not_send_imgs.set()
                        # there will be a function which will basically wait 
                        # for the whole thing to finish and then start again
                        pepper.not_send_imgs.clear()

        self.do_movement.clear()
