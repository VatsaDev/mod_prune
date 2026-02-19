#!/bin/bash

# CONFIG
export WANDB_PROJECT="clean_evo_prune_sN" # change N to what ever CKPT you are testing (ex. I went 2.2b->1.8b (s1), 1.8b->1.4b (s2), 1.4b->1.2b (s3), 1.2b->1b (s4))
export SWIFT_PATCH_CONV3D=1 # speed fix I found

BASE_CHECKPOINT="" # put your testing model here

TRAIN_DATASET="../../ani/kv_150k_data/kv_batch_0_150000/train.json" # 90/10 val split I made, just do the same with SFT data and drop it here
VAL_DATASET="../../ani/kv_150k_data/kv_batch_0_150000/val.json"

TEXT_TO_TEST=(1 2 3 4 5 6 7 8 9 10 11) # tests blocks 1-11, put whatever here, fair warning, in 0-N, 0 and N are a waste of time, input output layers are really well tuned and not redundant
VIS_TO_TEST=(1 2 3 4 5 6 7 8 9 10 11)

run_experiment() {
    local type=$1   # "text" or "vis"
    local idx=$2
    local RUN_NAME="rm_${type}_L${idx}"
    local TEMP_MODEL="temp_models/${RUN_NAME}" # intermediate pruning spot
    local RUN_OUTPUT="checkpoints/evo/${RUN_NAME}" # intermediate ckpt spot

    echo "##############################################"
    echo "TESTING REMOVAL OF ${type} LAYER ${idx}"
    echo "##############################################"

    # prune
    if [ "$type" == "text" ]; then
        python3 auto_clipper.py --model "$BASE_CHECKPOINT" --out "$TEMP_MODEL" --text_idx "$idx"
    else
        python3 auto_clipper.py --model "$BASE_CHECKPOINT" --out "$TEMP_MODEL" --vis_idx "$idx"
    fi

    # SFT
    export WANDB_NAME="$RUN_NAME"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    NPROC_PER_NODE=8 \
    swift sft \
        --model "$TEMP_MODEL" \
        --model_type qwen3_vl \
        --template qwen3_vl \
        --dataset ${TRAIN_DATASET} \
        --val_dataset ${VAL_DATASET} \
        --train_type full \
        --use_liger true \ # this works for speed during SFT but breaks GKD loss for some reason, I think its because the kernel fuses some output stage we need in GKD, needs debug ig
        --freeze_llm false \
        --freeze_vit false \
        --freeze_aligner false \
        --max_steps 100 \ # increase as nesc, in my case loss/acc both visibly maxxed out within 100 steps
        --learning_rate 5e-5 \
        --lr_scheduler_type cosine \
        --warmup_steps 20 \
        --save_steps 100 \
        --acc_strategy token \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --max_length 5000 \
        --torch_dtype bfloat16 \
        --output_dir "$RUN_OUTPUT" \
        --report_to wandb \
        --ddp_backend nccl \
        --ddp_find_unused_parameters true

    # cleanup to save disk space, probs dont need, but I'm used to it
    rm -rf "$TEMP_MODEL"
}

# all layer testing

for i in "${TEXT_TO_TEST[@]}"; do
    run_experiment "text" "$i"
done

for i in "${VIS_TO_TEST[@]}"; do
    run_experiment "vis" "$i"
done
