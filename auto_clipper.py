# same thing as other prune script, just a way for your bash scripts to call upon any arbitrary pruning

import torch
import argparse
import os
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

def prune_one_layer(model_id, output_path, text_idx, vis_idx):
    print(f"Loading {model_id} for pruning...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # text
    if text_idx != -1:
        layers = model.model.language_model.layers
        print(f"Removing Text Layer Index: {text_idx}")
        model.model.language_model.layers = torch.nn.ModuleList([l for i, l in enumerate(layers) if i != text_idx])

    # vision
    if vis_idx != -1:
        blocks = model.model.visual.blocks
        print(f"Removing Vision Block Index: {vis_idx}")
        model.model.visual.blocks = torch.nn.ModuleList([b for i, b in enumerate(blocks) if i != vis_idx])

    # configs
    model.config.text_config.num_hidden_layers = len(model.model.language_model.layers)
    model.config.vision_config.depth = len(model.model.visual.blocks)
    if hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = len(model.model.language_model.layers)

    # added a deepstack non-existent block check, for auto knockout, this may be biasing the model towards text tho
    max_vis = len(model.model.visual.blocks) - 1
    model.config.vision_config.deepstack_visual_indexes = [min(x, max_vis) for x in [2, 8, 14]] # NOTE hardcoded vals to my model, edit for your model before running

    print(f"Saving pruned model to {output_path}")
    model.save_pretrained(output_path, safe_serialization=False)
    processor.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--text_idx", type=int, default=-1)
    parser.add_argument("--vis_idx", type=int, default=-1)
    args = parser.parse_args()
    prune_one_layer(args.model, args.out, args.text_idx, args.vis_idx)
