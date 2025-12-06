#!/bin/bash

# Build image 
echo "--- Building Docker image: decentralized-llm-thesis:latest ---"
docker build -t decentralized-llm-thesis:latest .

# Without using cache
# docker build --no-cache -t decentralized-llm-thesis:latest .