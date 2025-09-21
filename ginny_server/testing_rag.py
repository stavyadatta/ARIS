from reasoner import Reasoner
from executor import Executor


class TestRag():
    def __init__(self):
        pass

    def __call__(self, face_id: str, transcription: str, is_rag: bool=True):
        person_details = Reasoner(transcription, face_id)
