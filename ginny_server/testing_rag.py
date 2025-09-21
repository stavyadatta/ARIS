from time import perf_counter
from reasoner import Reasoner
from executor import Executor


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
        for response_chunk in response:
            response_text = response_chunk.textchunk
            printed_chars += len(response_text)
            chunks += 1
            print(response_text, end="", flush=True)
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
    face_id = "face_1"
    is_rag = False

    test_rag = TestRag()
    test_rag(face_id, transcription, is_rag=is_rag)

