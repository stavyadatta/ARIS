#!/usr/bin/env bash
# Regenerate Python gRPC stubs from grpc.proto.
# Run this after editing grpc.proto. Requires: pip install grpcio-tools
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. grpc.proto
echo "Regenerated grpc_pb2.py and grpc_pb2_grpc.py in $HERE"
