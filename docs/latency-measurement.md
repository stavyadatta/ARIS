# Measuring iris_server turn latency

How the G1 turn path was instrumented, how it is measured, and what the
measurements have shown so far. Written so a result can be reproduced or
contradicted rather than taken on trust.

## Why this exists

Before this work the request path had no instrumentation at all — the only
`time.time()` under `media_manager`, `reasoner`, `executor` or `apis` built a
filename. Every claim about where a turn's seconds went was therefore a
model-latency estimate. The first real measurements contradicted two of them
immediately, so the standing rule is: **measure before optimising, and
re-measure after.**

## The metric

**Time to first chunk (TTFA)** — wall-clock from the server entering
`ProcessAudioImg` to the first `TextChunk` leaving it. This is what a person
standing in front of the robot experiences, and it is not the same as total
RPC time: work done after the reply is on the wire costs the person nothing.

Reporting both is what makes a *reordering* distinguishable from a *deletion*.
If TTFA falls while total RPC time holds, work moved off the reply path; if
both fall, work was removed.

## Instrumentation

`iris_server/turn_timing.py`. One turn is one `ProcessAudioImg` RPC.

- `turn(label)` — wraps the RPC, emits the breakdown when it ends
- `span(name)` — times a section; nests, and reads parent-before-child
- `mark(name, value)` — a non-timing fact (route taken, reply length)
- `record_first_moment(name)` — how far into the turn something *first*
  happened; a per-chunk call site therefore records time-to-first, not
  time-to-last

It lives at the package root, not under `utils/`, because every layer imports
it while `utils/__init__` constructs the Neo4j driver on import.

Two properties worth knowing:

- **Measurement never breaks a turn.** With no active turn every entry point
  is a no-op; emission failures are printed, not raised.
- **The active turn is thread-local.** gRPC serves each RPC on its own worker
  and drives that request's response generator on the same thread. A
  consequence that matters: *work deferred to a background thread disappears
  from the turn timeline*. That is correct — it is no longer on the critical
  path — but it means "the span vanished" is itself a result to check.

Output goes to stdout as `[timing]` blocks and appends one JSON object per
turn to `iris_server/logs/turn_timing.jsonl` (gitignored).

## Running an experiment

```bash
export IRIS_LATENCY_WORKDIR=/some/scratch   # holds audio/ and timing_client.py
test/iris_server/run_latency_experiment.sh <label> <utterance> <turns>
```

The script restarts the server on the current working tree, runs a fixed
number of turns against a fixed audio/image pair, and saves the JSONL under
the label. Fixtures used so far: utterances generated with OpenAI `tts-1`
(so they are stable and repeatable, unlike a live microphone), and
`database/face_db/face_1.png` as the frame so recognition always resolves.

### Protocol

1. **Discard the first turn.** Cold CUDA contexts and the first HTTPS
   connection make it unrepresentative. It is worth reporting separately —
   that is how the model warm-up work was justified — but never pooled with
   the rest.
2. **Choose the route deliberately.** They behave nothing alike. The gesture
   route exits on a regex gate with a constant reply and is the low-variance
   control (TTFA spread ~±20 ms). The speak route's TTFA is dominated by
   reply length and swings by seconds; raw medians across conditions are close
   to meaningless on it.
3. **Prefer a variance-independent statistic** when the route is noisy. For
   the deferred-persist change the useful measure was the residual inside
   `api_response` that `context_build + open_stream + generation` do not
   account for — that is exactly where the persist sat, and it does not move
   with reply length.
4. **Check side effects actually still happen.** A latency win that silently
   drops writes is not a win. Count the affected nodes in Neo4j before and
   after.

### Known confounders

- **The harness force-removes the container** (`docker rm -f`) as soon as the
  last reply lands. Anything deferred to a background thread can be killed
  mid-flight. This produced a real false negative once — see below.
- **Network variance to OpenAI is large** and unrelated to any code change.
  The same `speaking.persist` work measured 730 ms and 965 ms across runs.
- **Repeating one utterance** makes any text-keyed cache look better than it
  would in production. Vary the input before trusting a caching result.
- **Turns write to the live Neo4j.** Each speak or gesture turn adds two
  `Message` nodes to the person's chain, which then feed `get_last_k_msgs` on
  later turns.

## Results so far

Hardware: RTX 4090, Neo4j over LAN, OpenAI over WAN. Medians.

### Baseline

| route | TTFA | shape |
|---|---|---|
| `speak` | 9234 ms | 4169 ms gpt-4-turbo TTFT, 2222 ms generation, 863 ms retrieval, 724 ms persist |
| `g1 wave` | 1015 ms | 730 ms of it persisting a constant string |

Local (non-LLM) stages total ~190 ms: Whisper 138 ms, face 13 ms, decode
0.7 ms, Neo4j person lookup 5 ms, regex gates 0.0 ms.

### Changes measured

| change | metric | before | after |
|---|---|---|---|
| Warm models at startup | first-turn `resolve_face_id` | 1656.7 ms | 12.5 ms |
| | first-turn `whisper.transcribe` | 621.9 ms | 278.8 ms |
| Gesture: yield before persist | gesture TTFA | 1015 ms | 284 ms |
| Speak: persist after reply | `api_response` residual | 724.4 ms | 5.0 ms |
| Reply brevity (prompt) | long-route reply length | 455 chars | 94 chars |
| | long-route TTFA | 12395 ms | 7195 ms |
| | client mic-blanking | 30350 ms | 7588 ms |

Only ~2.0 s of the warm-up's 5.9 s first-turn improvement is attributable to
the change; the rest was a drop in the first OpenAI call that warming local
models cannot explain, and is not claimed.

### The false negative, recorded because it nearly misled us

After deferring the spoken-turn persist, Neo4j showed 7 new messages for 8
turns — one apparently lost. The change looked unsafe. Re-running with a
five-second settle before teardown produced 3 writes for 3 turns and no logged
errors, which located the cause in the harness rather than the code: the
daemon thread was being killed by `docker rm -f`. The residual exposure is
genuine but bounded — one in-flight turn at abrupt shutdown — and the gesture
route, which persists inside the RPC, lost nothing across the same 8 turns.

## The client is half the pipeline

`unitree-g1-edu` settles two things the server cannot:

- **The client buffers too.** `g1_client_cpp/aris_image_queue_smoke.cpp`
  accumulates `complete_reply += chunk.text()` across the whole stream and
  speaks only after `Finish()`. Server-side streaming alone would therefore
  change nothing.
- **G1's TTS cannot queue.** `TtsMaker` returns when the request is accepted,
  and per `docs/g1-client-cpp-guide.md` "submitting another sentence
  immediately can interrupt the first one". Sentence-at-a-time playback would
  have to be paced by a sleep against an unreliable duration estimate.

**Reply length is therefore the dominant client-side cost.** `robot/speak.cpp`
blanks the microphone for `clamp(chars x 77ms, 800, 30000) + 350` after
playback starts, so every character is paid for twice -- once generating it
behind the buffer, once waiting it out. A 455-character reply cost the full
30 s clamp.

Any latency work on a spoken turn should therefore report reply length
alongside the timings, and ideally the derived blanking figure, because that
term dwarfs everything on the server.

## Open questions

- gpt-4-turbo TTFT (4169 ms) is the single largest item on a spoken turn and
  is untouched by everything above; `speaking.py:80` pins the model.
- The remaining dead air is the generation tail, which the whole-utterance
  buffer in `_g1_conversation_chunks` pins to the *last* token. Removing it
  needs a wire change and a G1 client that can accept appended speech.
- Reply brevity comes from the prompt, not `max_tokens` -- the inherited 500
  never bound (the longest reply observed was 455 characters, ~114 tokens) and
  a cap low enough to bind truncates mid-sentence. Retrieved history also acts
  as few-shot examples, so a prompt change takes a few turns to converge.
- The route mix in real use is still unknown. `executor.py` logs the selected
  API and `turn_timing` records the route, so a day of real traffic would
  settle which of these numbers actually matters.
