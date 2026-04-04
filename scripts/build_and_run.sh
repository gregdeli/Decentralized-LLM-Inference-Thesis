#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Build image 
echo "--- Building Docker image: decentralized-llm-thesis:latest ---"
NO_CACHE_FLAG="$1"

if [ "$NO_CACHE_FLAG" == "--no-cache" ]; then
    docker build --no-cache -t decentralized-llm-thesis:latest .
else
    docker build -t decentralized-llm-thesis:latest .
fi

# Create and run containers 
echo "--- Starting Docker Compose services ---"
docker compose up