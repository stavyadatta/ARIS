# Request planner evaluation

Measures the planner that turns one spoken sentence into a gesture queue, a
refusal ("unsupported") or no action ("chat"). `cases.json` holds 100 labelled
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

Set `IRIS_PLANNER_MODEL` (or `OPENAI_MODEL`) to try another model; the model name
is printed and saved, the key never is. Calls run on 8 threads.
