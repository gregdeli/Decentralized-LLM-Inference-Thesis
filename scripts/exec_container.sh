#!/bin/bash

# 1. Get the container suffix and remove it from the argument list
CONTAINER_SUFFIX="$1"
shift

# 2. Check for the debug flag. If present, set variable and remove it from the list
DEBUG_FLAG=""
if [ "$1" == "-d" ]; then
    DEBUG_FLAG="-d"
    shift
fi

# 3. Get the Python file and remove it from the list
PYTHON_FILE="$1"
shift

# 4. Any remaining arguments in "$@" now belong to the Python script

CONTAINER_NAME="decentralized-llm-inference-thesis-${CONTAINER_SUFFIX}"

# -it -> interactive 
if [ "$DEBUG_FLAG" == "-d" ]; then
    docker exec -it "$CONTAINER_NAME" python -Xfrozen_modules=off -m debugpy --listen 0.0.0.0:5678 "$PYTHON_FILE" "$@"
else
    docker exec -it "$CONTAINER_NAME" python "$PYTHON_FILE" "$@"
fi

# Usage
# ./scripts/exec_container.sh client-1 distributed_chatbot.py --stream
# ./scripts/exec_container.sh client-1 client_gui.py
# ./scripts/exec_container.sh client-1 -d client_gui.py