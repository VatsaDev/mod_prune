#!/bin/bash

BASE="checkpoints/base"
ADAPTER_1="checkpoints/adapter"
FINAL="checkpoints/final"

echo "Merging Adapter into Base..."
swift export \
    --model "$BASE" \
    --adapters "$ADAPTER_1" \
    --merge_lora true \
    --output_dir "$FINAL"

echo "Final merged model is in: $FINAL"
