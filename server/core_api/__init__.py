from .face_recognition import _FaceRecognition
from .whisper2text import _WhisperSpeech2Text
from .yolo import _PersonDetectorCropper
from .chatgpt import _OpenAIHandler
from .grok import _GrokHandler
from .relationship_checker import _RelationshipChecker
from .attribute_finder import _AttributeFinder

FaceRecognition = _FaceRecognition()
WhisperSpeech2Text = _WhisperSpeech2Text()
PersonDetectionCropper = _PersonDetectorCropper()
ChatGPT = _OpenAIHandler()
Grok = _GrokHandler()
RelationshipChecker = _RelationshipChecker()
AttributeFinder = _AttributeFinder()

__all__ = ["FaceRecognition", 
           "WhisperSpeech2Text", 
           "PersonDetectionCropper", 
           "ChatGPT", 
           "Grok",
           "RelationshipChecker",
           "AttributeFinder",
           ]


