# iris_server smoke tests

Three plain-script suites covering the Iris (Unitree G1) server. They are
scripts rather than pytest cases because the server image ships no pytest.

```sh
test/iris_server/run_tests.sh                 # all three suites
test/iris_server/run_tests.sh test_pipeline   # one suite
```

The runner mounts the checkout over the image's baked-in copy, so it tests
working-tree code without a rebuild. It prefers `iris-server:latest` and falls
back to `ginny-server:latest`.

| file | covers |
| --- | --- |
| `test_pipeline.py` | transcription, face-id resolution, reasoner routing and gesture gates, the G1 reply/action contract, `StreamImages`/`GetBbox`/`ClearQueue`/`ProcessAudioImg` |
| `test_llm_handlers.py` | the ChatGPT and Grok handlers' shared vision plumbing and the methods the server calls on them |
| `test_action_queue.py` | the G1 action queue: gesture ordering and sequencing words, the `actions` array of the reply contract, the gesture API and the voice step; stubs Neo4j, so it needs no database |
| `harness.py` | import-path setup, core_api stubbing, pass/fail bookkeeping |

## Requirements

Neo4j must be reachable for the other suites: `utils/__init__.py` connects at
import time. `test_action_queue` stubs the
client (`harness.stub_neo4j_database`) and runs without it. The password is
read from `.env` at the repo root.

No API keys are needed — the model layer is stubbed, and the LLM handlers are
only inspected, never called.
