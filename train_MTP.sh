#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
MEGATRON_BIN="${VENV_DIR}/bin/megatron"
NVIDIA_SITE_PACKAGES="${VENV_DIR}/lib/python3.11/site-packages/nvidia"

export WANDB_PROJECT="${WANDB_PROJECT:-restore}"
export WANDB_NAME="${WANDB_NAME:-mtp_head1_steps1000_4xh200_ctx8192_mtp1_linear}"
export WANDB_ANONYMOUS="allow"
export SWIFT_MTP_ACCEPT_LEN_TARGET="${SWIFT_MTP_ACCEPT_LEN_TARGET:-16}"
export SWIFT_MTP_VAL_ACCEPT_LEN_SAMPLE_SIZE="${SWIFT_MTP_VAL_ACCEPT_LEN_SAMPLE_SIZE:-20}"
export SWIFT_MTP_TRAIN_ACCEPT_LEN_SAMPLE_SIZE="${SWIFT_MTP_TRAIN_ACCEPT_LEN_SAMPLE_SIZE:-${SWIFT_MTP_VAL_ACCEPT_LEN_SAMPLE_SIZE}}"

MODEL_DIR="${MODEL_DIR:-/root/checkpoints/best_48}"
LOAD_DIR="${LOAD_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/checkpoints/train_MTP_head1_steps1000_ctx8192_mtp1_linear}"
TRAIN_DATASET="${TRAIN_DATASET:-${SCRIPT_DIR}/../../data/train.json}"
VAL_DATASET="${VAL_DATASET:-${SCRIPT_DIR}/../../data/val.json}"
FINETUNE="${FINETUNE:-false}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SEED="${SEED:-42}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
TRAIN_ITERS="${TRAIN_ITERS:-1000}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-${TRAIN_ITERS}}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
EVAL_ITERS="${EVAL_ITERS:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
NUM_LAYERS="${NUM_LAYERS:-48}"
NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-4}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-auto}"
DECODER_FIRST_PIPELINE_NUM_LAYERS="${DECODER_FIRST_PIPELINE_NUM_LAYERS:-}"
DECODER_LAST_PIPELINE_NUM_LAYERS="${DECODER_LAST_PIPELINE_NUM_LAYERS:-}"

RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
RECOMPUTE_MODULES="${RECOMPUTE_MODULES:-core_attn}"

ATTN_IMPL="${ATTN_IMPL:-sdpa}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-false}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-unfused}"
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-true}"
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1}"
NO_LOAD_OPTIM="${NO_LOAD_OPTIM:-false}"
NO_LOAD_RNG="${NO_LOAD_RNG:-false}"

MAIN_GRADS_DTYPE="${MAIN_GRADS_DTYPE:-fp32}"
MAIN_PARAMS_DTYPE="${MAIN_PARAMS_DTYPE:-fp32}"
EXP_AVG_DTYPE="${EXP_AVG_DTYPE:-fp32}"
EXP_AVG_SQ_DTYPE="${EXP_AVG_SQ_DTYPE:-fp32}"
MODEL_PRECISION="${MODEL_PRECISION:-bf16}"
MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-auto}"

MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-1}"
MTP_LOSS_SCALING_FACTOR="${MTP_LOSS_SCALING_FACTOR:-0.3}"
MTP_LAYER_TYPE="${MTP_LAYER_TYPE:-}"
LR="${LR:-5e-6}"
MIN_LR="${MIN_LR:-5e-7}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.2}"
TRAIN_OUTPUT_EXAMPLE_STEPS="${TRAIN_OUTPUT_EXAMPLE_STEPS:-1}"
TRAIN_OUTPUT_EXAMPLE_MAX_CHARS="${TRAIN_OUTPUT_EXAMPLE_MAX_CHARS:-1024}"
TRAIN_OUTPUT_EXAMPLE_MAX_TOKENS="${TRAIN_OUTPUT_EXAMPLE_MAX_TOKENS:-256}"
LOG_TRAIN_OUTPUT_EXAMPLE="${LOG_TRAIN_OUTPUT_EXAMPLE:-true}"

export SWIFT_PATCH_CONV3D=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export OMP_NUM_THREADS=14

if [ ! -x "${MEGATRON_BIN}" ]; then
    echo "Missing megatron executable: ${MEGATRON_BIN}" >&2
    exit 1
fi

NVIDIA_LIB_PATHS=""
for libdir in "${NVIDIA_SITE_PACKAGES}"/*/lib; do
    if [ -d "${libdir}" ]; then
        if [ -n "${NVIDIA_LIB_PATHS}" ]; then
            NVIDIA_LIB_PATHS="${NVIDIA_LIB_PATHS}:${libdir}"
        else
            NVIDIA_LIB_PATHS="${libdir}"
        fi
    fi
done
if [ -n "${NVIDIA_LIB_PATHS}" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

LOAD_ARGS=()
if [ -n "${LOAD_DIR}" ]; then
    LOAD_ARGS+=(
        --load "${LOAD_DIR}"
        --no_load_optim "${NO_LOAD_OPTIM}"
        --no_load_rng "${NO_LOAD_RNG}"
    )
fi

MTP_LAYER_TYPE_ARGS=()
if [ -n "${MTP_LAYER_TYPE}" ]; then
    MTP_LAYER_TYPE_ARGS+=(--mtp_layer_type "${MTP_LAYER_TYPE}")
fi

MODEL_PRECISION="${MODEL_PRECISION,,}"
PRECISION_ARGS=()
case "${MODEL_PRECISION}" in
    bf16)
        PRECISION_ARGS+=(--torch_dtype bfloat16 --fp16 false --bf16 true)
        ;;
    fp16)
        PRECISION_ARGS+=(--torch_dtype float16 --fp16 true --bf16 false)
        ;;
    fp32)
        PRECISION_ARGS+=(--torch_dtype float32 --fp16 false --bf16 false)
        ;;
    *)
        echo "Unsupported MODEL_PRECISION: ${MODEL_PRECISION} (expected bf16, fp16, or fp32)" >&2
        exit 1
        ;;
esac

if [ "${MOE_GROUPED_GEMM}" = "auto" ]; then
    if [ "${MODEL_PRECISION}" = "bf16" ]; then
        MOE_GROUPED_GEMM="true"
    else
        MOE_GROUPED_GEMM="false"
    fi
fi

if [ "${SEQUENCE_PARALLEL}" = "auto" ]; then
    if [ "${TENSOR_MODEL_PARALLEL_SIZE}" -gt 1 ]; then
        SEQUENCE_PARALLEL="true"
    else
        SEQUENCE_PARALLEL="false"
    fi
fi

PIPELINE_LAYER_ARGS=()
if [ -n "${DECODER_FIRST_PIPELINE_NUM_LAYERS}" ]; then
    PIPELINE_LAYER_ARGS+=(--decoder_first_pipeline_num_layers "${DECODER_FIRST_PIPELINE_NUM_LAYERS}")
fi
if [ -n "${DECODER_LAST_PIPELINE_NUM_LAYERS}" ]; then
    PIPELINE_LAYER_ARGS+=(--decoder_last_pipeline_num_layers "${DECODER_LAST_PIPELINE_NUM_LAYERS}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
"${MEGATRON_BIN}" sft \
    "${LOAD_ARGS[@]}" \
    --model "${MODEL_DIR}" \
    --model_type qwen3_5 \
    --template qwen3_5 \
    --load_safetensors true \
    --save_safetensors true \
    --dataset "${TRAIN_DATASET}" \
    --val_dataset "${VAL_DATASET}" \
    --load_from_cache_file true \
    --seed "${SEED}" \
    --micro_batch_size "${MICRO_BATCH_SIZE}" \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --train_iters "${TRAIN_ITERS}" \
    --log_interval 1 \
    --recompute_granularity "${RECOMPUTE_GRANULARITY}" \
    --recompute_method "${RECOMPUTE_METHOD}" \
    --recompute_num_layers "${RECOMPUTE_NUM_LAYERS}" \
    --recompute_modules "${RECOMPUTE_MODULES}" \
    --no_gradient_accumulation_fusion true \
    --cross_entropy_loss_fusion true \
    --cross_entropy_fusion_impl native \
    --calculate_per_token_loss true \
    --use_flash_attn "${USE_FLASH_ATTN}" \
    --attention_backend "${ATTENTION_BACKEND}" \
    --attn_impl "${ATTN_IMPL}" \
    --optimizer adam \
    --optimizer_cpu_offload "${OPTIMIZER_CPU_OFFLOAD}" \
    --optimizer_offload_fraction "${OPTIMIZER_OFFLOAD_FRACTION}" \
    --use_precision_aware_optimizer true \
    --main_grads_dtype "${MAIN_GRADS_DTYPE}" \
    --main_params_dtype "${MAIN_PARAMS_DTYPE}" \
    --exp_avg_dtype "${EXP_AVG_DTYPE}" \
    --exp_avg_sq_dtype "${EXP_AVG_SQ_DTYPE}" \
    --dataloader_type cyclic \
    --manual_gc_interval 0 \
    --lr "${LR}" \
    --lr_decay_style cosine \
    --lr_decay_iters "${LR_DECAY_ITERS}" \
    --lr_warmup_iters 0 \
    --lr_warmup_fraction 0.05 \
    --min_lr "${MIN_LR}" \
    --weight_decay 0.1 \
    --clip_grad 1.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_eps 1e-08 \
    --sgd_momentum 0.9 \
    --repetition_penalty "${REPETITION_PENALTY}" \
    --save "${OUTPUT_DIR}" \
    --save_interval "${SAVE_INTERVAL}" \
    --no_save_optim true \
    --no_save_rng true \
    --finetune "${FINETUNE}" \
    --ckpt_format torch_dist \
    --no_initialization true \
    --auto_detect_ckpt_format true \
    --exit_on_missing_checkpoint true \
    --distributed_backend nccl \
    --use_distributed_optimizer true \
    --tensor_model_parallel_size "${TENSOR_MODEL_PARALLEL_SIZE}" \
    --pipeline_model_parallel_size "${PIPELINE_MODEL_PARALLEL_SIZE}" \
    --sequence_parallel "${SEQUENCE_PARALLEL}" \
    "${PIPELINE_LAYER_ARGS[@]}" \
    --context_parallel_size 1 \
    --distributed_timeout_minutes 300000 \
    --num_layers "${NUM_LAYERS}" \
    --hidden_size 5120 \
    --ffn_hidden_size 17408 \
    --num_attention_heads 24 \
    --group_query_attention true \
    --num_query_groups "${NUM_QUERY_GROUPS}" \
    --softmax_type vanilla \
    --max_position_embeddings 262144 \
    --position_embedding_type mrope \
    --mrope_section 11 11 10 \
    --rotary_base 10000000 \
    --rotary_percent 1.0 \
    --normalization RMSNorm \
    --norm_epsilon 1e-06 \
    --swiglu true \
    --glu_linear_offset 0.0 \
    --untie_embeddings_and_output_weights true \
    --disable_bias_linear true \
    --attention_dropout 0.0 \
    --hidden_dropout 0.0 \
    --kv_channels 256 \
    --qk_layernorm true \
    --transformer_impl transformer_engine \
    --moe_layer_freq 1 \
    --moe_router_topk 2 \
    --moe_router_dtype fp32 \
    --moe_router_score_function softmax \
    --moe_router_load_balancing_type aux_loss \
    --expert_model_parallel_size 1 \
    --expert_tensor_parallel_size 1 \
    --moe_token_dispatcher_type alltoall \
    --moe_grouped_gemm "${MOE_GROUPED_GEMM}" \
    --moe_aux_loss_coeff 0.0 \
    --moe_token_drop_policy probs \
    --kv_lora_rank 32 \
    --qk_head_dim 128 \
    --qk_pos_emb_head_dim 64 \
    --v_head_dim 128 \
    --mtp_num_layers "${MTP_NUM_LAYERS}" \
    --mtp_loss_scaling_factor "${MTP_LOSS_SCALING_FACTOR}" \
    "${MTP_LAYER_TYPE_ARGS[@]}" \
    --fp8_recipe delayed \
    --fp8_amax_history_len 1024 \
    --fp8_amax_compute_algo max \
    "${PRECISION_ARGS[@]}" \
    --attention_softmax_in_fp32 true \
    --tensorboard_log_interval 1 \
    --tensorboard_queue_size 50 \
    --log_timers_to_tensorboard true \
    --log_validation_ppl_to_tensorboard true \
    --log_memory_to_tensorboard true \
    --report_to wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_exp_name "${WANDB_NAME}" \
    --log_train_output_example "${LOG_TRAIN_OUTPUT_EXAMPLE}" \
    --train_output_example_steps "${TRAIN_OUTPUT_EXAMPLE_STEPS}" \
    --train_output_example_max_chars "${TRAIN_OUTPUT_EXAMPLE_MAX_CHARS}" \
    --train_output_example_max_tokens "${TRAIN_OUTPUT_EXAMPLE_MAX_TOKENS}" \
    --check_model false \
    --eval_interval "${EVAL_INTERVAL}" \
    --eval_iters "${EVAL_ITERS}" \
    --max_length "${SEQ_LENGTH}" \
    --seq_length "${SEQ_LENGTH}" \
    --num_workers "${NUM_WORKERS}" \
    --dataset_num_proc "${DATASET_NUM_PROC}" \
    --padding_free false
