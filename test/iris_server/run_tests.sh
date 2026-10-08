#!/usr/bin/env bash
#
# Run the iris_server smoke tests inside the server image.
#
# The suites need the image's dependencies (opencv, neo4j, openai) but none of
# its models, so they run in a throwaway container with the checkout mounted
# over the baked-in copy. Nothing is written to the repo.
#
# Usage:
#   test/iris_server/run_tests.sh                 # all suites
#   test/iris_server/run_tests.sh test_pipeline   # one suite
#
# NEO4J_PASSWORD comes from .env at the repo root; utils/__init__.py connects
# to Neo4j at import time, so Iris's own database must be up:
#   docker compose -f docker-compose.iris.yml up -d iris-neo4j

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IRIS_TEST_IMAGE:-iris-server:latest}"
SUITES=("${@:-test_pipeline test_llm_handlers test_action_queue test_turn_decision}")

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image $IMAGE not found; falling back to ginny-server:latest" >&2
    IMAGE="ginny-server:latest"
fi

# Compose strips the quotes around values in .env; docker run --env-file does
# not, so read the password out directly instead.
NEO4J_PASSWORD="$(grep '^NEO4J_PASSWORD' "$REPO_ROOT/.env" \
    | sed 's/^NEO4J_PASSWORD=//' | tr -d '"'\''\n')"

status=0
for suite in ${SUITES[*]}; do
    echo "===== $suite"
    docker run --rm --network host --entrypoint python \
        -e NEO4J_PASSWORD="$NEO4J_PASSWORD" \
        -e NEO4J_URL="${NEO4J_URL:-bolt://localhost:7688}" \
        -e OPENAI_API_KEY="${OPENAI_API_KEY:-unused-in-tests}" \
        -e GROK_API_KEY="${GROK_API_KEY:-unused-in-tests}" \
        -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-unused-in-tests}" \
        -e API_KEY="${API_KEY:-unused-in-tests}" \
        -e PYTHONPATH=/workspace:/workspace/grpc_communication:/workspace/test/iris_server \
        -v "$REPO_ROOT/iris_server:/workspace/iris_server:ro" \
        -v "$REPO_ROOT/test:/workspace/test:ro" \
        -w /workspace/test/iris_server \
        "$IMAGE" "$suite.py" || status=1
done

exit $status
