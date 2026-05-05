#!/bin/bash

# Compile all .proto files in this directory using Python gRPC tools

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m grpc_tools.protoc \
    -I"$SCRIPT_DIR" \
    --python_out="$SCRIPT_DIR" \
    --grpc_python_out="$SCRIPT_DIR" \
    "$SCRIPT_DIR"/*.proto

echo "Proto compilation complete."
