# Qwen Supports GQA, which already parameterizes MQA/MHA, (Q=KV, MHA, KV=1, MQA)
# this repo will let you cut the number of heads down to any num, always keep it to a power of 2 obv
# Qwen starts at 8, going 8->4->1 was the most stable route for me, 
# Kimi K2.5 has 

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
import os

# Config

old_mod = "checkpoints/m_11T_10V/v2-20260219-130058/checkpoint-4700"
new_mod = "checkpoints/qwen_1b_mqa_averaged"
new_heads = 1 # mqa, start old//2 or 4

def get_param_count(model):
    return sum(p.numel() for p in model.parameters()) / 1e9

def reduce_kv_heads_qwen_averaging(model_id, output_path, new_kv_heads=1):
  
    print(f"Loading model: {model_id}")
  
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )
  
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print(f"Initial Size: {get_param_count(model):.3f}B")

    text_config = model.config.text_config
    hidden_size = text_config.hidden_size
    num_q_heads = text_config.num_attention_heads
    num_kv_heads_old = text_config.num_key_value_heads
    head_dim = hidden_size // num_q_heads

    if num_kv_heads_old % new_kv_heads != 0:
        raise ValueError(f"Cannot divide {num_kv_heads_old} heads into {new_kv_heads} groups evenly.")

    group_size = num_kv_heads_old // new_kv_heads
    print(f"Collapsing {num_kv_heads_old} KV heads into {new_kv_heads} heads.")
    print(f"Averaging groups of {group_size} heads into 1.")

    # so we average heads together in this code, originally was just slicing tensors and keeping N heads, but avging N heads leads to more stability and better performance
    # perhaps avging is keeping the heads specializations accurate? 
  
    layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers

    for i, layer in enumerate(layers):
        attn = layer.self_attn

        for proj_name in ['k_proj', 'v_proj']:
            proj = getattr(attn, proj_name)

            # reshape weights to [num_kv_heads_old, head_dim, hidden_size]
            # proj weights are usually [out_dim, in_dim]
            old_weight = proj.weight.data
            w = old_weight.view(num_kv_heads_old, head_dim, hidden_size)

            # average groups: [new_kv_heads, group_size, head_dim, hidden_size] -> mean across group_size (2nd, so axis=1)
            new_w = w.view(new_kv_heads, group_size, head_dim, hidden_size).mean(dim=1)

            # reshape back to [new_kv_heads * head_dim, hidden_size]
            proj.weight = torch.nn.Parameter(new_w.reshape(new_kv_heads * head_dim, hidden_size).clone())

            # handle bias if it exists, dont think any models we use should, just covering
            if proj.bias is not None:
                old_bias = proj.bias.data
                b = old_bias.view(num_kv_heads_old, head_dim)
                new_b = b.view(new_kv_heads, group_size, head_dim).mean(dim=1)
                proj.bias = torch.nn.Parameter(new_b.reshape(new_kv_heads * head_dim).clone())

    # update configs
    model.config.text_config.num_key_value_heads = new_kv_heads
    if hasattr(model.config, "num_key_value_heads"):
        model.config.num_key_value_heads = new_kv_heads

    print(f"Final Size: {get_param_count(model):.3f}B")

    # save
    model.save_pretrained(output_path, safe_serialization=False)
    processor.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    if hasattr(model, 'generation_config'):
        model.generation_config.save_pretrained(output_path)

    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    reduce_kv_heads_qwen_averaging(
        model_id=old_mod,
        output_path=new_mod,
        new_kv_heads=new_heads
    )
