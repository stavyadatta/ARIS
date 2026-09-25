# iris_server smoke tests

Two plain-script suites covering the Iris (Unitree G1) server. They are
scripts rather than pytest cases because the server image ships no pytest.

```sh
test/iris_server/run_tests.sh                 # both suites
test/iris_server/run_tests.sh test_pipeline   # one suite
```

The runner mounts the checkout over the image's baked-in copy, so it tests
working-tree code without a rebuild. It prefers `iris-server:latest` and falls
back to `ginny-server:latest`.

| file | covers |
| --- | --- |
| `test_pipeline.py` | transcription, face-id resolution, reasoner routing and gesture gates, the G1 reply/action contract, `StreamImages`/`GetBbox`/`ClearQueue`/`ProcessAudioImg` |
| `test_llm_handlers.py` | the ChatGPT and Grok handlers' shared vision plumbing and the methods the server calls on them |
| `harness.py` | import-path setup, core_api stubbing, pass/fail bookkeeping |

## Requirements

Neo4j must be reachable: `utils/__init__.py` connects at import time, so every
suite needs it even when the test does not touch the database. The password is
read from `.env` at the repo root.

No API keys are needed — the model layer is stubbed, and the LLM handlers are
only inspected, never called.
