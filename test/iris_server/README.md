# iris_server smoke tests

Plain-script suites covering the Iris (Unitree G1) server. They are
scripts rather than pytest cases because the server image ships no pytest.

```sh
test/iris_server/run_tests.sh                 # all the suites
test/iris_server/run_tests.sh test_pipeline   # one suite
```

The runner mounts the checkout over the image's baked-in copy, so it tests
working-tree code without a rebuild. It prefers `iris-server:latest` and falls
back to `ginny-server:latest`.

| file | covers |
| --- | --- |
| `test_pipeline.py` | transcription, face-id resolution, the reasoner's confirmation-reply and thanks shortcuts, the G1 reply/action contract, `StreamImages`/`GetBbox`/`ClearQueue`/`ProcessAudioImg` |
| `test_llm_handlers.py` | the ChatGPT and Grok handlers' shared vision plumbing and the methods the server calls on them |
| `test_turn_decision.py` | the classifier decides whether a turn is physical and the planner builds the queue: the physical-state rule, each planner outcome (queue, refusal, no steps, failure), no planner call for chat, greetings and confirmations, the injectable classifier call, and that the old keyword gates are gone; every model is a fake |
| `test_action_queue.py` | the G1 action queue: the step planner (fake model), how a decision is routed, unsupported steps and their one fixed refusal wording, long queues, the `actions` array of the reply contract, the gesture API and the voice step; stubs Neo4j, so it needs no database |
| `live_action_planner_check.py` | OPT-IN, not run by `run_tests.sh`: sends five sentences to the real planner model and prints the answers; spends API credits and needs `OPENAI_API_KEY` |
| `planner_eval/` | the labelled-sentence evaluation of the planner alone and of the whole classifier-then-planner path; its offline checks run like the suites above but from that folder, see `planner_eval/README.md` |
| `harness.py` | import-path setup, core_api stubbing, pass/fail bookkeeping |

## Requirements

Neo4j must be reachable for the other suites: `utils/__init__.py` connects at
import time. `test_action_queue` stubs the
client (`harness.stub_neo4j_database`) and runs without it. The password is
read from `.env` at the repo root.

No API keys are needed — the model layer is stubbed, and the LLM handlers are
only inspected, never called.
