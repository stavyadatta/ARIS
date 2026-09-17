# Making Iris answer faster

How slow Iris was, what we changed, what each change was worth, and how we
measured it. Every number here was measured on the real server, not estimated.

---

## 1. The short version

A spoken turn went from about **9 seconds to about 7 seconds**. A long
"do you remember me" answer went from **12.4 seconds to 7.2 seconds**. A
misheard sentence went from **7.3 seconds of a confident wrong answer** to
**1.2 seconds of "sorry, say that again"**.

| Change | Impact |
|---|---|
| Made replies short | long answers: 12.4 s → 7.2 s |
| Save the turn to the database *after* replying | −0.7 s on every spoken turn |
| Gesture speaks before saving to the database | gestures: 1.0 s → 0.28 s |
| Misheard speech now says "I didn't catch that" | misheard turns: 7.3 s → 1.2 s |
| Warm up the AI models at startup | first turn after restart: 14.5 s → 8.6 s |
| Give up on a stuck OpenAI call after 20 s | removes a 4 s silence (was up to 10 min) |
| *(added later, on purpose)* gestures reply to what you said | gestures: 0.28 s → 1.6 s |

---

## 2. Why we measured before changing anything

There was no timing anywhere in the request path. Every belief about where the
seconds went was a guess. The first real measurements proved two guesses wrong
straight away:

- We assumed speech-to-text (Whisper) took 0.5–3 seconds. **It takes 0.14 s.**
- We never suspected the first turn after a restart. **It was 6 seconds slower
  than every turn after it.**

There was also a theory that the "reasoner" and "executor" stages should run at
the same time to save time. Measurement killed it: **the executor takes 0.1
milliseconds.** It is a dictionary lookup, not a stage. Running it in parallel
would save one ten-thousandth of the wait.

**Rule going forward: measure, change, measure again.**

---

## 3. How we measure

### What we count

**Time to first chunk** — from the moment the server receives the audio to the
moment it sends back the first piece of the reply. This is what a person
standing in front of the robot actually waits for.

We also record **total request time**, which includes work done *after* the
reply was sent. Recording both lets us prove the difference between:

- **moving** work out of the way (first-chunk time drops, total stays the same)
- **deleting** work (both drop)

### The tool

`iris_server/turn_timing.py` times every stage of every turn. It prints a block
like this, and appends the same data as JSON to
`iris_server/logs/turn_timing.jsonl`:

```
[timing] process_audio_img total=7309.0ms
[timing]   transcribe                        297.8ms   4.1%
[timing]     whisper.transcribe              261.9ms   3.6%
[timing]   resolve_face_id                    12.6ms   0.2%
[timing]   reason                            970.0ms  13.3%
[timing]     reasoner.classify_llm           965.1ms  13.2%
[timing]   api_response                     7940.1ms  86.4%
[timing]     speaking.context_build          862.6ms   9.4%
[timing]     speaking.open_stream           4169.0ms  45.4%
[timing]     speaking.generation            2222.4ms  24.2%
[timing] facts route=speak reply_chars=122 first_chunk_at_ms=7309
```

It can never break a turn: if no turn is active every call does nothing, and if
printing fails it says so rather than raising an error.

### Running a test

```bash
export IRIS_LATENCY_WORKDIR=/some/folder     # holds audio/ and timing_client.py
test/iris_server/run_latency_experiment.sh <name> <utterance> <number-of-turns>
```

This restarts the server using the current code, sends the same audio and photo
a fixed number of times, and saves the results under `<name>`.

Test audio is generated with OpenAI text-to-speech so every run says exactly the
same words. A live microphone would say something slightly different each time
and runs could not be compared. The photo is `database/face_db/face_1.png`, so
face recognition always succeeds.

### Four rules we follow

1. **Throw away the first turn.** The models are cold and it is not
   representative. Report it separately if it matters.
2. **Know which route you are testing.** A gesture turn and a spoken turn behave
   completely differently. Gestures are steady (±20 ms) and make a good control.
   Spoken turns swing by seconds depending on how long the reply is.
3. **When the numbers are noisy, measure something that is not.** For the
   database change in 5.3, reply length made the total jump around by seconds —
   so we measured only the gap the database work used to fill. That gap does not
   change with reply length.
4. **Check the side effects still happen.** A change that looks faster because
   it quietly stopped saving data is not an improvement. Count the database rows
   before and after.

---

## 4. Where the time went, originally

One spoken turn, measured:

| Stage | Time |
|---|---|
| Speech to text (Whisper) | 0.30 s |
| Face recognition | 0.01 s |
| Deciding what to do (LLM) | 0.97 s |
| Fetching past conversation | 0.86 s |
| **Waiting for the first word from gpt-4-turbo** | **4.17 s** |
| Generating the rest of the reply | 2.22 s |
| Saving the turn to the database | 0.72 s |

All the local work — speech-to-text, face recognition, image decoding, database
lookup — adds up to about **0.19 seconds**. Everything else is waiting on OpenAI
or on the database.

---

## 5. The changes

### 5.1 Replies were far too long

**Before.** The prompt told Iris to answer "do you remember me" with the
person's name *"along with their shared experiences"*, and the example in the
prompt listed several. Iris obeyed, producing 455 characters reciting the
person's degree, thesis, hobbies and reading list.

**What we changed.** The instruction and its example now ask for the name and
**one** shared detail, turned back into a question.

**Impact.**

| | Before | After |
|---|---|---|
| Reply length | 455 characters | 94 characters |
| Generating the reply | 3.39 s | 0.68 s |
| Time to first chunk | 12.40 s | 7.20 s |

**How we tested.** Six turns of "What do you remember about me?" before and
after, same audio. We also read the replies back out of the database to confirm
they still ended in a complete sentence.

**What did not work.** Our first attempt set a hard limit of 60 tokens. The
numbers improved, but replies came out **cut off mid-sentence** — one ended on
"How's your". The robot would have said that out loud. It turned out the
original 500-token limit was never being reached anyway (the longest reply ever
recorded was about 114 tokens), so the limit was never the problem. Shortness
has to come from the prompt. The limit is now 120, purely as a safety net.

**Worth knowing.** Iris reads its own past replies back as examples. Right after
this change it still produced a few long ones, copying its own history. It
settled after a handful of turns.

### 5.2 A gesture waited on the database before speaking

**Before.** When Iris waves, the reply is short and already decided. But the code
saved the turn to the database *before* handing the reply over — two calls to
OpenAI and one database write, all while the person waited.

**What we changed.** Send the reply first, then save. The save still finishes
inside the same request; only its position moved.

**Impact.** Gesture time to first chunk: **1.02 s → 0.28 s (−72%)**.

**How we tested.** Six turns before, seven after. The two sets do not overlap at
all — the slowest "after" turn was still faster than the fastest "before" turn.
Crucially, **total request time went up** by a similar amount, which proves the
work moved rather than disappeared.

### 5.3 A spoken reply waited on the database too

**Before.** The same problem on spoken turns, but the fix above does not work
here: the reply is produced word by word and the code holds all the words until
the end, so anything after the last word still lands in front of the person.

**What we changed.** The save now runs on a background thread.

**Impact.** **0.72 s → 0.005 s** removed from the waiting time.

**How we tested.** Reply length made total time swing between 3.8 s and 9.7 s,
so totals were useless for comparison. Instead we measured only the leftover gap
where the saving used to sit, which does not depend on reply length. Before:
0.72 s. After: 0.005 s. No overlap between runs.

**The trade-off, measured rather than assumed.** A turn killed mid-save is lost.
Eight turns where we killed the server immediately saved seven. Three turns
given five seconds to finish saved three, with no errors. So the exposure is one
turn, and only if the server is killed at that exact moment.

**A mistake worth recording.** When we first saw "7 saves for 8 turns" we
believed the change was losing data. It was not — our own test script was
killing the server too fast. We only found out by re-running with a delay.
**If a result looks alarming, suspect the test before the code.**

### 5.4 Misheard speech was answered as if understood

**Before.** When Iris could not make out what was said, it correctly decided
"bad input" — and then the code **threw that decision away** and reused whatever
Iris was doing last, usually "have a conversation". So the noise went to the
conversation model, which answered from memory and produced a confident reply to
something nobody said. In practice it repeated whichever line appeared most
often in that person's history.

**What we changed.** "Bad input" now reaches the code written to handle it,
which produces the "I didn't catch that" reply and the head-scratch gesture.
Both had been unreachable.

**Impact.** A misheard turn: **~7.3 s of a wrong answer → ~1.2 s of "sorry,
could you say that again"**.

**How we tested.** Three turns of silent audio. The log now shows
`Person State: bad input`, and Iris replies *"I am still learning to listen in a
noisy room. Please say that again when you are ready."*

### 5.5 The models were cold on the first turn

**Before.** Speech-to-text and face recognition each set themselves up on first
use. That cost landed on whoever spoke first after a restart — usually a demo
audience.

**What we changed.** Run one throwaway recognition of each at startup, before
the server accepts any request.

**Impact.** First turn after restart: **14.5 s → 8.6 s**.

| | Before | After |
|---|---|---|
| Face recognition, first turn | 1.66 s | 0.013 s |
| Speech to text, first turn | 0.62 s | 0.28 s |

**Honesty note.** Only about **2 seconds** of that 5.9 s improvement is ours.
The rest was the first connection to OpenAI being slow, which warming up local
models cannot explain. We are not claiming it.

**One catch.** Warming up with a blank image was not enough. Face *detection*
starts up on any image, but face *recognition* only starts up once a face has
actually been found — worth 1.2 s on its own. We warm it with a fake face so it
does not depend on the face database containing anyone.

### 5.6 A failed OpenAI call cost 4 seconds of silence

**Before.** The OpenAI client used default settings: wait up to **10 minutes**,
retry twice. Worse, when a call failed the code returned the error as ordinary
text instead of raising it — so the backup provider (Grok) never ran. The turn
ended with the robot saying **nothing at all**.

**What we changed.** Wait at most 20 seconds, retry once, and let failures raise
so the backup can actually take over.

**Impact.** No change to normal speed. Removes a 4-second silence when OpenAI
fails, and caps the worst case.

**How we tested.** We hit this for real — the API key ran out of credit
mid-testing and we captured the exact crash.

### 5.7 The one change that made things slower, on purpose

Gestures used to speak one fixed sentence. Now they answer what was actually
said:

> *"I just passed my thesis defense, can you give me a high five?"*
> → *"Congratulations, Tavia, you absolutely nailed it!"*

**Impact.** Gesture time to first chunk: **0.28 s → 1.63 s** (0.81 s of that is
the extra AI call). A real cost, chosen deliberately. It is still faster than
the 1.02 s this path cost before any of this work.

**A mistake worth recording.** Our first version showed the AI the old fixed
sentence "for tone". It copied it back word for word — so we spent 1.3 seconds
regenerating a constant. Removing the example fixed it immediately.

---

## 6. What we did not fix, and why

**Waiting 4.17 s for gpt-4-turbo's first word.** The single biggest remaining
item — **57% of a spoken turn**. It is one word in the code (the model name in
`speaking.py`). Changing the model was ruled out, so it stands. It is larger
than everything in section 5 put together.

**Sending the reply piece by piece as it is written.** Iris holds the whole
reply until it is finished. Sending it in pieces would let the robot start
talking sooner. We investigated and decided against it:

- The robot's own software **also** holds the whole reply before speaking, so
  changing only the server would achieve nothing.
- The robot's speech system **cannot queue sentences** — sending a second one
  interrupts the first. Its own documentation says so.
- The most it could save is about 0.95 s, and it needs coordinated changes in
  two codebases.

Shortening replies (5.1) was cheaper and gained more.

---

## 7. Things that caught us out

- **Whisper is fast.** We assumed 0.5–3 s; it is 0.14 s, even on silence. We
  nearly optimised something worth 3% of the turn.
- **A limit that never applies does nothing.** The 500-token reply limit was
  never reached, so lowering it looked free — until it began cutting sentences
  in half.
- **Showing the AI an example makes it copy the example.** This happened twice.
- **Our own test script invented a bug.** See 5.3.
- **Network noise is large.** The same database work measured 0.73 s and 0.97 s
  on different runs. Effects smaller than that cannot be found by comparing
  totals.
- **Iris imitates its own history.** Both prompt changes took several turns to
  take hold, because Iris was copying its own older replies.

---

## 8. The robot side matters more than the server

Two settings in the robot's own software (`unitree-g1-edu`, not this repo) cost
more than everything in this document combined:

- It waited **1.29 s of silence** before deciding you had stopped talking.
  Typical for this kind of system is 0.5–0.8 s. Now 0.69 s.
- After speaking it stops listening for a **guessed** duration — 77 ms per
  character. On a 455-character reply that was **30 seconds deaf**. Shortening
  replies (5.1) cut this to about 7 s; retuning the guess to 67 ms/char cut it
  further.

That second number is still an estimate. It can be measured exactly: time the
robot speaking a sentence of known length and set the constant from that.

---

## 9. Still open

- Nobody knows the real mix of gesture, spoken and misheard turns in normal use.
  Every turn now records its route, so a day of real use would settle which of
  these numbers actually matters.
- Iris sometimes says it cannot see a person who has not moved. Undiagnosed. The
  log line `[face_id] none ... recognized_votes=[...]` was added for exactly this
  and will show whether camera frames were rejected or simply not recognised.
- Iris and the Pepper robot share one database. Pepper's old replies appear in
  Iris's memory and Iris sometimes copies them. Not fixed, because cleaning it
  would damage Pepper's memory too.
