#!/usr/bin/env bash
# Restart the iris server on the current working tree, run a fixed set of
# turns, and save the per-turn timing JSONL under the given experiment label.
#
# Usage: IRIS_LATENCY_WORKDIR=/path/to/work run_latency_experiment.sh <label> <utterance> <turns>
set -euo pipefail

LABEL="$1"; UTTERANCE="$2"; TURNS="$3"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRATCH="${IRIS_LATENCY_WORKDIR:?set IRIS_LATENCY_WORKDIR to a directory holding audio/ fixtures and the client}"

set -a; . "$REPO/.env"; set +a

rm -f "$REPO/iris_server/logs/turn_timing.jsonl"
docker rm -f iris-timing >/dev/null 2>&1 || true

docker run -d --name iris-timing --network host --gpus all \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" -e NEO4J_PASSWORD="$NEO4J_PASSWORD" \
  -e GROK_API_KEY="$GROK_API_KEY" -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  -e API_KEY="$API_KEY" -e PYTHONUNBUFFERED=1 -e HF_HOME=/root/.cache/huggingface \
  -v "$REPO/iris_server":/workspace/iris_server \
  -v "$REPO/grpc_communication":/workspace/grpc_communication \
  -v "$REPO/display_imgs":/workspace/display_imgs \
  -v "$REPO/database":/workspace/database \
  -v "$SCRATCH":/client \
  -v aris_whisper-cache:/root/.cache/whisper \
  -v aris_insightface-cache:/root/.insightface \
  -v aris_torch-cache:/root/.cache/torch \
  -v aris_hf-cache:/root/.cache/huggingface \
  iris-server:latest python main.py >/dev/null

for _ in $(seq 1 90); do
  docker logs iris-timing 2>&1 | grep -q "gRPC server running" && break
  sleep 5
done

docker exec -w /workspace/iris_server iris-timing \
  python /client/timing_client.py "/client/audio/${UTTERANCE}.wav" \
  /workspace/database/face_db/face_1.png "$LABEL" "$TURNS" 2>&1 | grep ttfa

cp "$REPO/iris_server/logs/turn_timing.jsonl" "$SCRATCH/${LABEL}.jsonl"
docker rm -f iris-timing >/dev/null 2>&1 || true
echo "saved $SCRATCH/${LABEL}.jsonl"
