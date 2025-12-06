#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Build image 
echo "--- Building Docker image: decentralized-llm-thesis:latest ---"
docker build -t decentralized-llm-thesis:latest . # -t -> tag 

# Create and run containers 
echo "--- Starting Docker Compose services ---"
docker compose up