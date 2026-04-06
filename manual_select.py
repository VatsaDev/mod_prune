#!/usr/bin/env python3
# Manually select text and/or vision layers to prune from an HF checkpoint.

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText as AutoModelClass
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoModelClass


DEFAULT_MODEL = "checkpoints/your_model/"
DEFAULT_OUT = "checkpoints/out"
DEFAULT_TEXT_CHOP = [4, 10]
DEFAULT_VIS_CHOP = [9, 10]
DEFAULT_DEEPSTACK = [1, 5, 8]


def parse_indices(raw):
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return sorted({int(x) for x in raw})
    raw = raw.strip()
    if not raw:
        return []
    return sorted({int(part.strip()) for part in raw.split(",") if part.strip()})


def parse_one_based_indices(raw):
    indices = parse_indices(raw)
    if any(idx <= 0 for idx in indices):
        raise ValueError("One-based indices must be >= 1.")
    return [idx - 1 for idx in indices]


def get_param_count(model):
    return sum(p.numel() for p in model.parameters()) / 1e9


def filter_by_index(values, remove_idx):
    remove_idx = set(remove_idx)
    return [value for i, value in enumerate(values) if i not in remove_idx]


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def checkpoint_contains_mtp(model_path: Path) -> bool:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return False
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    return any(key.startswith("mtp.") for key in index.get("weight_map", {}))


def describe_removed_values(values, remove_idx, *, key):
    if values is None:
        return []
    remove_idx = set(remove_idx)
    return [{
        "zero_based_idx": idx,
        "one_based_idx": idx + 1,
        key: values[idx],
    } for idx in sorted(remove_idx) if 0 <= idx < len(values)]


def update_text_config(model, text_remove_idx):
    if not text_remove_idx:
        return

    new_num_layers = len(model.model.language_model.layers)
    text_config = model.config.text_config
    text_config.num_hidden_layers = new_num_layers

    filtered_layer_types = None
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is not None:
        filtered_layer_types = filter_by_index(layer_types, text_remove_idx)
        text_config.layer_types = filtered_layer_types

    if hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = new_num_layers

    if hasattr(model.model.language_model, "config"):
        lang_config = model.model.language_model.config
        if lang_config is not text_config:
            lang_config.num_hidden_layers = new_num_layers
            if filtered_layer_types is not None:
                lang_config.layer_types = list(filtered_layer_types)


def update_vision_config(model, vis_remove_idx, new_deepstack_idx):
    if not vis_remove_idx:
        return

    new_depth = len(model.model.visual.blocks)
    vision_config = model.config.vision_config
    vision_config.depth = new_depth

    if new_deepstack_idx:
        deepstack = list(new_deepstack_idx)
    else:
        deepstack = list(getattr(vision_config, "deepstack_visual_indexes", []))

    max_vis = new_depth - 1
    if max_vis >= 0:
        deepstack = [min(idx, max_vis) for idx in deepstack]
    else:
        deepstack = []
    vision_config.deepstack_visual_indexes = deepstack

    if hasattr(model.model.visual, "config"):
        visual_config = model.model.visual.config
        if visual_config is not vision_config:
            visual_config.depth = new_depth
            visual_config.deepstack_visual_indexes = list(deepstack)


def sync_auxiliary_args(model_path, output_path, model):
    source_args_path = model_path / "args.json"
    if not source_args_path.exists():
        return

    with source_args_path.open(encoding="utf-8") as handle:
        args = json.load(handle)

    text_config = getattr(model.config, "text_config", model.config)
    layer_types = getattr(text_config, "layer_types", None)
    num_layers = len(model.model.language_model.layers)

    targets = [args]
    extra_args = args.get("extra_args")
    if isinstance(extra_args, dict):
        targets.append(extra_args)

    for target in targets:
        target["num_layers"] = num_layers
        if layer_types is not None:
            target["layer_types"] = list(layer_types)

    write_json(output_path / "args.json", args)


def prune_mod(model_id, output_path, text_remove_idx, vis_remove_idx, new_deepstack_idx):
    model_path = Path(model_id).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if output_path.exists():
        raise FileExistsError(f"Output path already exists: {output_path}")
    if checkpoint_contains_mtp(model_path):
        raise ValueError(
            "Checkpoint contains MTP tensors. "
            "Use prune_text_layer_safetensors.py to preserve mtp.* weights when pruning.")

    print(f"Loading model from {model_path}")
    model = AutoModelClass.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    print(f"Initial Size: {get_param_count(model):.3f}B")

    source_text_layer_types = getattr(model.config.text_config, "layer_types", None)
    removed_text_layers = describe_removed_values(
        source_text_layer_types, text_remove_idx, key="layer_type")
    removed_vision_layers = describe_removed_values(
        list(range(len(model.model.visual.blocks))), vis_remove_idx, key="original_block_idx")

    if text_remove_idx:
        lang_layers = model.model.language_model.layers
        print(f"Removing Text Layer Indices: {text_remove_idx}")
        model.model.language_model.layers = torch.nn.ModuleList(
            [layer for i, layer in enumerate(lang_layers) if i not in set(text_remove_idx)]
        )
        update_text_config(model, text_remove_idx)

    if vis_remove_idx:
        vis_blocks = model.model.visual.blocks
        print(f"Removing Vision Block Indices: {vis_remove_idx}")
        model.model.visual.blocks = torch.nn.ModuleList(
            [block for i, block in enumerate(vis_blocks) if i not in set(vis_remove_idx)]
        )
        update_vision_config(model, vis_remove_idx, new_deepstack_idx)

    print(f"Final Size: {get_param_count(model):.3f}B")
    print(
        f"Layers: {len(model.model.language_model.layers)} Text / "
        f"{len(model.model.visual.blocks)} Vis"
    )

    output_path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output_path, safe_serialization=True)
    sync_auxiliary_args(model_path, output_path, model)
    processor.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    if hasattr(processor, "image_processor"):
        processor.image_processor.save_pretrained(output_path)
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        processor.video_processor.save_pretrained(output_path)
    if hasattr(model, "generation_config"):
        model.generation_config.save_pretrained(output_path)

    # Preserve auxiliary multimodal/tokenizer assets that save_pretrained may skip.
    for filename in [
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "merges.txt",
        "vocab.json",
        "special_tokens_map.json",
    ]:
        src = model_path / filename
        dst = output_path / filename
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    manifest = {
        "source_model": str(model_path),
        "output_path": str(output_path),
        "text_remove_idx": list(text_remove_idx),
        "text_remove_layer_numbers": [idx + 1 for idx in text_remove_idx],
        "removed_text_layers": removed_text_layers,
        "vis_remove_idx": list(vis_remove_idx),
        "vis_remove_layer_numbers": [idx + 1 for idx in vis_remove_idx],
        "removed_vision_layers": removed_vision_layers,
        "vision_deepstack_visual_indexes": list(
            getattr(model.config.vision_config, "deepstack_visual_indexes", [])
        ),
        "remaining_text_layers": len(model.model.language_model.layers),
        "remaining_vision_layers": len(model.model.visual.blocks),
    }
    write_json(output_path / "prune_manifest.json", manifest)

    print(f"Saved pruned model to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument(
        "--text-chop",
        type=str,
        default=",".join(str(x) for x in DEFAULT_TEXT_CHOP),
        help="Comma-separated zero-based text layer indices to prune.",
    )
    parser.add_argument(
        "--text-chop-one-based",
        type=str,
        default=None,
        help="Comma-separated one-based text layer numbers to prune.",
    )
    parser.add_argument(
        "--vis-chop",
        type=str,
        default=",".join(str(x) for x in DEFAULT_VIS_CHOP),
        help="Comma-separated zero-based vision block indices to prune.",
    )
    parser.add_argument(
        "--vis-chop-one-based",
        type=str,
        default=None,
        help="Comma-separated one-based vision block numbers to prune.",
    )
    parser.add_argument(
        "--deepstack",
        type=str,
        default=",".join(str(x) for x in DEFAULT_DEEPSTACK),
        help="Comma-separated deepstack indices for vision after pruning.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.text_chop_one_based is not None and args.text_chop != ",".join(str(x) for x in DEFAULT_TEXT_CHOP):
        raise ValueError("Use either --text-chop or --text-chop-one-based, not both.")
    if args.vis_chop_one_based is not None and args.vis_chop != ",".join(str(x) for x in DEFAULT_VIS_CHOP):
        raise ValueError("Use either --vis-chop or --vis-chop-one-based, not both.")
    prune_mod(
        model_id=args.model,
        output_path=args.out,
        text_remove_idx=parse_one_based_indices(args.text_chop_one_based)
        if args.text_chop_one_based is not None else parse_indices(args.text_chop),
        vis_remove_idx=parse_one_based_indices(args.vis_chop_one_based)
        if args.vis_chop_one_based is not None else parse_indices(args.vis_chop),
        new_deepstack_idx=parse_indices(args.deepstack),
    )
