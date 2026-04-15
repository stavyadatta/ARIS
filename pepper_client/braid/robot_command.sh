#!/usr/bin/env bash
# Launch the BRAID pepper client.
#
# Runs from /workspace so that the `grpc_communication` and `pepper_client`
# top-level packages resolve for imports inside braid_client.py.
#
# Mirrors /workspace/pepper_client/robot_command.sh (the speaker_client
# launcher) but targets the BRAID closed-loop pipeline instead.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${HERE}/../.." && pwd)"
cd "${WORKSPACE}"

# Make sure /workspace is on PYTHONPATH for module-style imports.
export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"

python -m pepper_client.braid.braid_client \
    --braid \
    --robot-ip 192.168.0.52 \
    --robot-port 9559 \
    --server 172.27.72.27:50051 \
    --listen-ip 192.168.0.50 \
    --listen-port 52100 \
    --tick-seconds 30 \
    --num-ticks 0 \
    --log-level INFO \
    "$@"
