#!/bin/bash

# HTTP API Evaluation Script for PRGB
# Usage: ./run_http_eval.sh <API_URL> <MODEL_NAME> <DATA_PATH> <OUTPUT_PATH> <API_KEY>

API_URL=${1}
MODEL_NAME=${2:-"DeepSeek-R1-0528"}
DATA_PATH=${3:-"data/zh.jsonl"}
OUTPUT_PATH=${4:-"./results"}
API_KEY_ARG=${5:-""}

# Set API key as environment variable if provided as argument
if [ -n "$API_KEY_ARG" ]; then
    export API_KEY="$API_KEY_ARG"
    echo "API key set from command line argument: $API_KEY"
elif [ -n "$(printenv API_KEY)" ]; then
    echo "Using existing API_KEY environment variable"
else
    echo "Warning: No API key provided. Please set API_KEY environment variable or pass it as the 5th argument."
    echo "You can set it with: export API_KEY=your_api_key"
fi

echo "Starting HTTP API evaluation..."
echo "API URL: $API_URL"
echo "Model Name: $MODEL_NAME"
echo "Data Path: $DATA_PATH"
echo "Output Path: $OUTPUT_PATH"
echo "API KEY: $API_KEY"

python eval.py \
    --model-name "$MODEL_NAME" \
    --model-path "$API_URL" \
    --data-path "$DATA_PATH" \
    --output-path "$OUTPUT_PATH" \
    --batch-size 8 \
    --temperature 0.7 \
    --inference-mode False 