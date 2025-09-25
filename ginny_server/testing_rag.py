from time import perf_counter
from reasoner import Reasoner
from executor import Executor

# Speaking.get_time() can be used to get the time taken, however once you 
# get time make sure you call Speaking.reset_time() so that the time on Speaking side 
# can be reset
from apis import Speaking

class TestRag:
    def __init__(self):
        pass

    def __call__(self, face_id: str, transcription: str, is_rag: bool = True):
        # Build reasoner input
        person_details = Reasoner(transcription, face_id)

        # --- Timing: Executor creation ---
        t0 = perf_counter()
        person_details.is_rag = is_rag
        response = Executor(person_details)
        t1 = perf_counter()

        # --- Timing: iterate + print loop ---
        printed_chars = 0
        chunks = 0
        t2 = perf_counter()
        text = ""
        for response_chunk in response:
            response_text = response_chunk.textchunk
            printed_chars += len(response_text)
            chunks += 1
            print(response_text, end="", flush=True)
            text += response_text
        t3 = perf_counter()

        print("\n")  # final newline after stream

        # --- Convert to ms and report ---
        exec_init_ms = (t1 - t0) * 1000.0
        stream_print_ms = (t3 - t2) * 1000.0
        total_ms = (t3 - t0) * 1000.0

        print(
            f"[Timing]\n"
            f"  Executor setup: {exec_init_ms:.2f} ms\n"
            f"  Stream + print loop: {stream_print_ms:.2f} ms\n"
            f"  Total (setup + loop): {total_ms:.2f} ms\n"
            f"[Stats] chunks={chunks}, printed_chars={printed_chars}"
        )


if __name__ == "__main__":
    transcription = "Who let the dogs out GINNY?"
    face_id = ["face_369", "face_6", "face_1"]
    is_rag = [True, False]

    for face in face_id:
        for rag_bool in is_rag:
            print(f"\n#### THIS IS FOR {face} and rag is {rag_bool}#####")
            test_rag = TestRag()
            test_rag(face, transcription, is_rag=rag_bool)

