from reasoner import Reasoner
from executor import Executor


class TestRag():
    def __init__(self):
        pass

    def __call__(self, face_id: str, transcription: str, is_rag: bool=True):
        person_details = Reasoner(transcription, face_id)

        response = Executor(person_details)
        mode = 'default'

        for response_chunk in response:
            response_text = response_chunk.textchunk
            print(response_text, end='', flush=True)

        print("\n")


if __name__ == "__main__":
    transcription = "Hey there GINNY, how are you doing"
    face_id = "face_1"

    test_rag = TestRag()
    test_rag(face_id, transcription)
