#!/bin/bash

NO_CACHE_FLAG="$1"

if [ "$NO_CACHE_FLAG" == "--no-cache" ]; then
    docker build --no-cache -t decentralized-llm-thesis:latest .
else
    docker build -t decentralized-llm-thesis:latest .
fi

# Build image 
# echo "--- Building Docker image: decentralized-llm-thesis:latest ---"
# docker build -t decentralized-llm-thesis:latest .

# Without using cache
# docker build --no-cache -t decentralized-llm-thesis:latest .