from .face_recognition import _FaceRecognition
from .whisper2text import _WhisperSpeech2Text
from .llama import _Llama
from .claude import _ClaudeImageProcessor
from .yolo import _PersonDetectorCropper, _YOLODetector
from .chatgpt import _OpenAIHandler
from .grok import _GrokHandler
from .relationship_checker import _RelationshipChecker
from .attribute_finder import _AttributeFinder
from .clip_classification import _ClipClassification
from .trackers import OCSort

FaceRecognition = _FaceRecognition()
WhisperSpeech2Text = _WhisperSpeech2Text()
Llama = _Llama()
PersonDetectionCropper = _PersonDetectorCropper()
YOLODetector = _YOLODetector()
Claude = _ClaudeImageProcessor()
ChatGPT = _OpenAIHandler()
Grok = _GrokHandler()
RelationshipChecker = _RelationshipChecker()
AttributeFinder = _AttributeFinder()
ClipClassification = _ClipClassification()

__all__ = ["FaceRecognition", 
           "WhisperSpeech2Text", 
           "Llama", 
           "PersonDetectionCropper", 
           "YOLODetector", 
           "Claude", 
           "ChatGPT", 
           "Grok",
           "RelationshipChecker",
           "AttributeFinder",
           "ClipClassification"
           ]


