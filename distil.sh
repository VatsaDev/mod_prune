#!/bin/bash

# full distillation
# your model learning the output of the 8b RL

ulimit -n 1048576 # dont remember exact cause of this one, but it used to cause bugs midrun

export IMAGE_MAX_TOKEN_NUM=2048 # might increase

export HF_HUB_ENABLE_HF_TRANSFER=1
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200
export TORCH_NCCL_ENABLE_MONITORING=0
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

export WANDB_API_KEY=YOUR_KEY
export WANDB_PROJECT=kv_distillation

export SWIFT_PATCH_CONV3D=1 # speed fix

STUDENT_MODEL="clean_build/checkpoints/m_11T_10V/v3-20260219-150354/checkpoint-5700" 
TEACHER_MODEL="../models/kv_8b_rl_14k_0125/"
TRAIN_DATASET="../ani/kv_150k_data/kv_batch_0_150000/train.json"
VAL_DATASET="../ani/kv_150k_data/kv_batch_0_150000/val.json"
OUTPUT_DIR="./checkpoints/qwen_1b_8b"

# 2 GPU student VLLM

# sudo fuser kill 8000/tcp between Runs, it and VLLM server hang at times
# make sure to match max model lens, otherwise you can have OOM crashes mid-run

echo "Starting VLLM server on GPU 0-1..."
CUDA_VISIBLE_DEVICES=0,1 \
swift rollout \
    --model_type qwen3_vl \
    --template qwen3_vl \
    --use_hf true \
    --port 8000 \ 
    --model ${STUDENT_MODEL} \
    --vllm_tensor_parallel_size 2 \
    --vllm_max_model_len 3000 & 

VLLM_PID=$!
echo "VLLM server started with PID: $VLLM_PID"

# taken from orig modal notebook
echo "Waiting 120 seconds for VLLM server to initialize..."
sleep 120

# 6 GPU GKD training

# originally colocate is best single node, but vis fix solved latency issues and keeping it server based helps for future multi-node scaling

echo "Starting GKD training on GPUs 2-7..."
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type gkd \
    --use_hf true \
    --model ${STUDENT_MODEL} \
    --teacher_model ${TEACHER_MODEL} \
    --dataset ${TRAIN_DATASET} \
    --val_dataset ${VAL_DATASET} \
    --load_from_cache_file false \
    --model_type qwen3_vl \
    --template qwen3_vl \
    --seq_kd false \
    --lmbda 1 \
    --beta 1 \
    --dataset_shuffle true \
    --acc_strategy token \
    --freeze_vit false \
    --freeze_aligner false \
    --freeze_llm false \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --train_type full \
    --use_logits_to_keep true \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --save_only_model true \
    --save_total_limit 20 \
    --logging_steps 5 \
    --save_steps 100 \
    --eval_steps 100 \
    --output_dir ${OUTPUT_DIR} \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.025 \
    --dataloader_num_workers 16 \
    --max_completion_length 2048 \
    --vllm_max_model_len 3000 \
    --temperature 0.4 \
    --top_p 0.9 \
    --strict false \
    --overlong_filter true \
    --repetition_penalty 1.2 \
    --deepspeed zero1 \
    --teacher_deepspeed zero2 \
    --report_to wandb

# sometimes doesnt work, double check nvidia-smi and fuser
echo "Training completed. Cleaning up VLLM server..."
kill $VLLM_PID
