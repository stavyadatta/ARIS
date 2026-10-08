# Request planner evaluation

Measures what the robot does with one spoken sentence: a gesture queue, a
refusal ("unsupported") or no action ("chat"). There are two stages. The
planner-only stage (`run_planner_eval.py`) asks just the planner that builds the
queue. The end-to-end stage (`run_end_to_end_eval.py`) asks the per-turn
classifier first, which decides whether the sentence is a physical request at
all, and only then the planner, exactly as the robot does. `cases.json` holds 100 labelled
sentences in ten categories (ordered and long queues, repeats, unsupported or
mixed requests, chat that only mentions gestures, noisy speech-to-text,
prompt-injection attempts, ambiguous requests). Labels follow the plain meaning
of each sentence; ambiguous or debatable ones carry a `note`.

Scoring is exact match: the same kind and, for a queue, the same ordered
actions. The most important number is the FALSE-ACTION rate, chat sentences the
planner turned into a queue or a refusal. A planner failure other than "no
physical step requested" is reported as `could_not_plan` and counts as wrong.

## Run

Offline checks (no key, no network, no robot, no database):

    cd /mnt/SSD2/minh/ARIS && docker run --rm --entrypoint python -e OPENAI_API_KEY=x -e NEO4J_PASSWORD=unused -e PYTHONPATH=/workspace:/workspace/grpc_communication:/workspace/test/iris_server -v $PWD/iris_server:/workspace/iris_server:ro -v $PWD/test:/workspace/test:ro -w /workspace/test/iris_server/planner_eval iris-server:latest test_planner_eval.py

Add `run_planner_eval.py --fake` instead for a runner dry run (every answer is
the expected label, so accuracy must print 100%).

Live run (about 100 real model requests, costs credits). `--env-file` hands the
key to the container without printing it:

    cd /mnt/SSD2/minh/ARIS && docker run --rm --entrypoint python --env-file .env -e PYTHONPATH=/workspace:/workspace/grpc_communication:/workspace/test/iris_server -v $PWD/iris_server:/workspace/iris_server:ro -v $PWD/test:/workspace/test:ro -w /workspace/test/iris_server/planner_eval iris-server:latest run_planner_eval.py

The mounts are read-only, so results go to `/tmp/planner_eval_results.json`
inside the container and vanish with it. To keep them, add
`-v /tmp/planner_eval:/out` before the image name and `--out /out/results.json`
after `run_planner_eval.py`.

## End-to-end stage

`run_end_to_end_eval.py` builds the server's own `_Reasoner` with the two model
calls swapped in and asks its `decide` method about every sentence, so the
classifier prompt, the physical-state rule and the routing under test are the
production code. The classifier call has no Grok fallback here, so a failure is
measured, not hidden. It reports everything the planner-only stage does, plus:
physical cases the classifier missed, chat cases it wrongly flagged, how many of
those the planner vetoed (answered "no steps requested"), the end-to-end
false-action rate, turns where the classifier asked "did you mean?", turns where
the planner failed and the classifier's own state was kept, and latency per turn
split into turns that skipped the planner and turns that went through it.

Offline checks of the evaluation itself (fakes only):

    cd /mnt/SSD2/minh/ARIS && docker run --rm --entrypoint python -e OPENAI_API_KEY=x -e NEO4J_PASSWORD=unused -e PYTHONPATH=/workspace:/workspace/grpc_communication:/workspace/test/iris_server -v $PWD/iris_server:/workspace/iris_server:ro -v $PWD/test:/workspace/test:ro -w /workspace/test/iris_server/planner_eval iris-server:latest test_end_to_end_eval.py

Runner dry runs (same command with `run_end_to_end_eval.py --fake`, accuracy must
print 100%; add `--fake-mistakes` and the fake classifier misses three physical
cases and over-flags three chat ones, one of which also fools the fake planner,
so the figures for those can be seen).

Live run (about 100 classifier and 70 planner requests, costs credits).
`docker run --env-file` keeps literal quotes around values, so give it a
temporary file with the quotes stripped, and delete it afterwards. Do not print
`.env`:

    grep '^OPENAI_API_KEY=' /mnt/SSD2/minh/ARIS/.env | tr -d "\"'" > /tmp/iris_eval.env && chmod 600 /tmp/iris_eval.env
    mkdir -p /tmp/e2e_out
    cd /mnt/SSD2/minh/_wt/aris-classifier-trigger && docker run --rm --entrypoint python --env-file /tmp/iris_eval.env -e PYTHONPATH=/workspace:/workspace/grpc_communication:/workspace/test/iris_server -v $PWD/iris_server:/workspace/iris_server:ro -v $PWD/test:/workspace/test:ro -v /tmp/e2e_out:/out -w /workspace/test/iris_server/planner_eval iris-server:latest run_end_to_end_eval.py --out /out/e2e_results.json
    rm /tmp/iris_eval.env

Pass `--cases /workspace/test/iris_server/planner_eval/cases_holdout.json` (same
format as `cases.json`) to score a second set. Both runners print a model
error as its class name and HTTP status only, never its message, because an API
error can echo a masked key.

Set `IRIS_PLANNER_MODEL` (or `OPENAI_MODEL`) to try another model; the model name
is printed and saved, the key never is; `OPENAI_MODEL` sets the classifier's model. Calls run on 8 threads.
