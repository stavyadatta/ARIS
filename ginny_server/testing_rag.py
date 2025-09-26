from time import perf_counter
import csv
from pathlib import Path

from reasoner import Reasoner
from executor import Executor
from apis import Speaking  # Speaking.get_time(), Speaking.reset_time()

QUESTIONS = [
    "Based on your memory, what do you already know about me? If you have specifics, summarise them; otherwise give a general, privacy-safe answer.",
    "What are my main interests or hobbies you’ve observed? If you know concrete details, list them; otherwise provide a general response.",
    "Which topics do I most often discuss with you? Use specifics if available; otherwise reply in general terms.",
    "How would you describe my communication style? Cite examples if you remember; otherwise provide a general description.",
    "What preferences of mine (tools, formats, tone) do you recall? If unsure, answer generally and note the uncertainty.",
    "What is my profession or area of study/work, according to your memory? If unknown, provide a general, careful answer.",
    "What ongoing projects of mine can you summarise from past interactions? If none are known, state that and answer generally.",
    "What do you know about my friends or colleagues I have mentioned? If you recall names/roles, summarise carefully; otherwise stay general.",
    "How would you characterise my relationships with those people? Use only remembered facts; if unknown, answer in general terms.",
    "If you lack specific memories about me, what responsible, general guidance would you give someone like me about next steps or resources?"
]

# ==== FILL THESE 10 ENTRIES ====
# message_len is your current count of Message nodes (or tokens/messages) for that face.
face_profiles = [
    {"face_id": "face_1",   "message_len": 1104},
    {"face_id": "face_6",   "message_len": 151},
    {"face_id": "face_60",   "message_len": 152},
    {"face_id": "face_218",   "message_len": 116},
    {"face_id": "face_63",   "message_len": 120},
    {"face_id": "face_162",   "message_len": 106},
    {"face_id": "face_197",   "message_len": 80},
    {"face_id": "face_365",   "message_len": 92},
    {"face_id": "face_192",   "message_len": 54},
    {"face_id": "face_224",  "message_len": 66},
    {"face_id": "face_271",  "message_len": 20},
    {"face_id": "face_277",  "message_len": 16},
    {"face_id": "face_318",  "message_len": 46},
    {"face_id": "face_377",  "message_len": 28},
]
# ==============================

OUT_CSV = Path("rag_benchmark_results.csv")
PRINT_STREAM = True  # set True if you also want to see streamed chunks in console

class TestRag:
    def __call__(self, face_id: str, transcription: str, is_rag: bool = True):
        # Build reasoner input
        person_details = Reasoner(transcription, face_id)

        # --- Timing: Executor creation ---
        t0 = perf_counter()
        person_details.is_rag = is_rag
        # Reset Speaking timing before the run to isolate each call
        Speaking.reset_time()
        response = Executor(person_details)
        t1 = perf_counter()

        # --- Timing: iterate stream ---
        printed_chars = 0
        chunks = 0
        t2 = perf_counter()
        text = ""
        for response_chunk in response:
            response_text = response_chunk.textchunk
            printed_chars += len(response_text)
            chunks += 1
            if PRINT_STREAM:
                print(response_text, end="", flush=True)
            text += response_text
        t3 = perf_counter()
        if PRINT_STREAM:
            print("\n")

        # --- Convert to ms ---
        exec_init_ms = (t1 - t0) * 1000.0
        stream_print_ms = (t3 - t2) * 1000.0
        total_ms = (t3 - t0) * 1000.0

        # Speaking time (ms) from TTS side, then reset to avoid spillover
        speaking_ms = Speaking.get_time()
        Speaking.reset_time()

        return {
            "exec_init_ms": round(exec_init_ms, 2),
            "stream_ms": round(stream_print_ms, 2),
            "total_ms": round(total_ms, 2),
            "speaking_ms": round(speaking_ms, 2) if speaking_ms is not None else None,
            "chunks": chunks,
            "printed_chars": printed_chars,
            "response_text": text.strip(),
        }

def main():
    # CSV header
    fieldnames = [
        "face_id",
        "message_len",
        "question_idx",
        "question",
        "is_rag",
        "exec_init_ms",
        "stream_ms",
        "total_ms",
        "speaking_ms",
        "chunks",
        "printed_chars",
        "response_text",
    ]

    tester = TestRag()
    rows = []

    for face in face_profiles:
        fid = face["face_id"]
        mlen = face["message_len"]

        for q_idx, q in enumerate(QUESTIONS, start=1):
            for rag_bool in (True, False):
                if PRINT_STREAM:
                    print(f"\n#### face={fid} | RAG={rag_bool} | Q{q_idx}: {q} ####")
                result = tester(fid, q, is_rag=rag_bool)
                rows.append({
                    "face_id": fid,
                    "message_len": mlen,
                    "question_idx": q_idx,
                    "question": q,
                    "is_rag": rag_bool,
                    **result
                })

    # write CSV
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()

