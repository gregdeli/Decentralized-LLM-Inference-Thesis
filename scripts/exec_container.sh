#!/bin/bash

CONTAINER_SUFFIX="$1"
PYTHON_FILE="$2"
DEBUG_FLAG="$3"

CONTAINER_NAME="decentralized-llm-inference-thesis-${CONTAINER_SUFFIX}"

# -it -> interactive 
if [ "$DEBUG_FLAG" == "-d" ]; then
    docker exec -it "$CONTAINER_NAME" python -Xfrozen_modules=off -m debugpy --listen 0.0.0.0:5678 "$PYTHON_FILE"
else
    docker exec -it "$CONTAINER_NAME" python "$PYTHON_FILE"
fi

# Usage:
# ./scripts/exec_container.sh client-1 client_gui.py         # normal run
# ./scripts/exec_container.sh client-1 client_gui.py -d      # run with debugpy