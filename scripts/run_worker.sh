#!/bin/bash

# Parse command-line input arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--model-path) 
            MODEL_PATH="$2"
            shift 2
            ;;
        -q|--quantize) 
            QUANTIZE="$2"
            shift 2
            ;;
        -l|--log-level) 
            LOG_LEVEL="$2"
            shift 2
            ;;
        -n|--num-layers) 
            NUM_LAYERS="$2"
            shift 2
            ;;
        -h|--help) 
            echo "Usage: $0 [options] <worker-name>"
            echo "Options:"
            echo "  -m, --model-path   Set the model path"
            echo "  -q, --quantize     Set quantization flag 0 or 1"
            echo "  -l, --log-level    Set log level"
            echo "  -n, --num-layers   Set number of layers"
            echo "Example:"
            echo "  $0 -q 1 -n 8 worker-B"
            exit 0
            ;;
        -*) 
            echo "Unknown option passed: $1"
            exit 1 
            ;;
        *) 
            # If it doesn't start with a dash, it is the positional argument for the worker name
            WORKER_NAME="$1"
            shift 1
            ;;
    esac
done

# Execute the container with the configured parameters
MODEL_PATH="$MODEL_PATH" \
QUANTIZE="$QUANTIZE" \
LOG_LEVEL="$LOG_LEVEL" \
NUM_LAYERS="$NUM_LAYERS" \
docker compose up -d "$WORKER_NAME"