# Held-out planner evaluation set

`cases_holdout.json` has 50 labelled requests, written separately from the first 100 cases.
It checks that the prompts were not tuned to those 100 cases.
Run it ONCE, after tuning is finished. Never use it to tune, and do not add or edit cases afterwards.
Labels come from the plain meaning of each sentence; `note` marks the debatable ones.
Counts: multi_queue 13, long_queue 4, repeat 3, single 5, unsupported 6,
chat_gesture_words 11, small_talk 3, noisy 3, injection 2.
By kind: queue 29, unsupported 6, chat 15.
