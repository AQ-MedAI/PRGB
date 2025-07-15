MODEL_PATH=${1}
MODEL_NAME=${2:-"Qwen3_infer"}
DATA_PATH=${3:-"data/zh.jsonl"}
OUTPUT_PATH=${4:-"./results"}

python eval.py \
    --model-name "$MODEL_NAME" \
    --model-path "$MODEL_PATH" \
    --data-path "$DATA_PATH" \
    --output-path "$OUTPUT_PATH" \
    --batch-size 8
