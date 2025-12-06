#!/bin/bash

CONTAINER_SUFFIX="$1"
PYTHON_FILE="$2"

CONTAINER_NAME="decentralized-llm-inference-thesis-${CONTAINER_SUFFIX}"

# -it -> interactive 
docker exec -it "$CONTAINER_NAME" python -m debugpy --listen 0.0.0.0:5678  "$PYTHON_FILE"

# To run client:
# ./scripts/exec_container.sh client-1 distributed_chatbot.py