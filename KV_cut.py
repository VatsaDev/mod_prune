#!/usr/bin/env python3
"""Reduce KV heads in Qwen checkpoints while preserving MTP tensors.

This operates directly on safetensors shards so checkpoints that carry
`mtp.*` weights are not silently stripped by an HF save_pretrained round-trip.
Only full-attention decoder layers are rewritten; linear-attention layers are
left untouched because they do not carry `self_attn.{k,v}_proj` tensors.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE_KV_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.self_attn\.(k_proj|v_proj)\.(weight|bias)$"
)
MTP_KV_RE = re.compile(r"^mtp\.layers\.(\d+)\.self_attn\.(k_proj|v_proj)\.(weight|bias)$")


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def copy_auxiliary_files(source_dir: Path, output_dir: Path, skip_names: set[str]):
    for src in sorted(source_dir.iterdir()):
        if src.name in skip_names:
            continue
        if src.name.startswith("iter_"):
            continue
        if src.name in {"latest_checkpointed_iteration.txt", "latest_wandb_artifact_path.txt"}:
            continue
        if src.suffix == ".safetensors":
            continue
        if src.is_dir():
            shutil.copytree(src, output_dir / src.name)
        else:
            shutil.copy2(src, output_dir / src.name)


def get_full_attention_layers(config_payload: dict) -> set[int]:
    text_config = config_payload.get("text_config", config_payload)
    layer_types = text_config.get("layer_types")
    if not isinstance(layer_types, list):
        raise ValueError("Expected config to contain text_config.layer_types.")
    return {
        idx for idx, layer_type in enumerate(layer_types)
        if layer_type in {"full_attention", "attention"}
    }


def should_prune_key(key: str, full_attention_layers: set[int]):
    match = BASE_KV_RE.match(key)
    if match is not None:
        layer_idx = int(match.group(1))
        if layer_idx in full_attention_layers:
            return True, f"decoder:{layer_idx}", match.group(2), match.group(3)
        return False, None, None, None

    match = MTP_KV_RE.match(key)
    if match is not None:
        mtp_idx = int(match.group(1))
        return True, f"mtp:{mtp_idx}", match.group(2), match.group(3)

    return False, None, None, None


def reduce_kv_tensor(tensor: torch.Tensor, *, old_heads: int, new_heads: int) -> torch.Tensor:
    if tensor.ndim not in (1, 2):
        raise ValueError(f"Unsupported tensor rank for KV reduction: {tensor.ndim}")
    if tensor.shape[0] % old_heads != 0:
        raise ValueError(f"Leading dim {tensor.shape[0]} is not divisible by old_heads={old_heads}")

    head_dim = tensor.shape[0] // old_heads
    if old_heads % new_heads != 0:
        raise ValueError(f"Cannot divide {old_heads} heads into {new_heads} groups evenly.")
    group_size = old_heads // new_heads

    if tensor.ndim == 2:
        hidden_size = tensor.shape[1]
        view = tensor.view(old_heads, head_dim, hidden_size)
        reduced = view.view(new_heads, group_size, head_dim, hidden_size).mean(dim=1)
        return reduced.reshape(new_heads * head_dim, hidden_size).clone()

    view = tensor.view(old_heads, head_dim)
    reduced = view.view(new_heads, group_size, head_dim).mean(dim=1)
    return reduced.reshape(new_heads * head_dim).clone()


def update_config(config_payload: dict, *, new_kv_heads: int):
    text_config = config_payload.get("text_config", config_payload)
    text_config["num_key_value_heads"] = new_kv_heads
    if "num_key_value_heads" in config_payload:
        config_payload["num_key_value_heads"] = new_kv_heads
    return config_payload


def update_args(args_payload: dict, *, new_kv_heads: int):
    def _patch(target):
        if not isinstance(target, dict):
            return
        if "num_query_groups" in target:
            target["num_query_groups"] = new_kv_heads
        if "num_key_value_heads" in target:
            target["num_key_value_heads"] = new_kv_heads

    _patch(args_payload)
    _patch(args_payload.get("extra_args"))
    return args_payload


def build_manifest(
    *,
    source_model: Path,
    output_path: Path,
    old_kv_heads: int,
    new_kv_heads: int,
    full_attention_layers: set[int],
    modified_tensors: list[str],
    preserved_mtp_keys: int,
):
    return {
        "source_model": str(source_model),
        "output_path": str(output_path),
        "old_num_key_value_heads": old_kv_heads,
        "new_num_key_value_heads": new_kv_heads,
        "full_attention_layer_numbers": [idx + 1 for idx in sorted(full_attention_layers)],
        "modified_tensor_count": len(modified_tensors),
        "modified_tensors": modified_tensors,
        "preserved_mtp_keys": preserved_mtp_keys,
    }


def reduce_kv_heads_qwen_averaging(model_path: Path, output_path: Path, *, new_kv_heads: int):
    model_path = model_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Output path already exists: {output_path}")

    source_index = load_json(model_path / "model.safetensors.index.json")
    source_config = load_json(model_path / "config.json")
    source_args = load_json(model_path / "args.json") if (model_path / "args.json").exists() else None

    text_config = source_config.get("text_config", source_config)
    old_kv_heads = int(text_config["num_key_value_heads"])
    if old_kv_heads == new_kv_heads:
        raise ValueError(f"Checkpoint already uses num_key_value_heads={new_kv_heads}.")
    if old_kv_heads % new_kv_heads != 0:
        raise ValueError(f"Cannot divide {old_kv_heads} heads into {new_kv_heads} groups evenly.")

    full_attention_layers = get_full_attention_layers(source_config)
    output_path.mkdir(parents=True, exist_ok=False)
    skip_names = {"model.safetensors.index.json", "config.json"}
    if source_args is not None:
        skip_names.add("args.json")
    copy_auxiliary_files(model_path, output_path, skip_names)

    shard_names = sorted(set(source_index["weight_map"].values()))
    weight_map = {}
    total_size = 0
    modified_tensors = []
    preserved_mtp_keys = 0

    for shard_name in shard_names:
        src_shard = model_path / shard_name
        tensors = {}
        metadata = None
        with safe_open(src_shard, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                should_prune, location, proj_name, tensor_kind = should_prune_key(
                    key, full_attention_layers
                )
                if should_prune:
                    tensor = reduce_kv_tensor(tensor, old_heads=old_kv_heads, new_heads=new_kv_heads)
                    modified_tensors.append(f"{location}:{proj_name}:{tensor_kind}")
                if key.startswith("mtp."):
                    preserved_mtp_keys += 1
                tensors[key] = tensor
                weight_map[key] = shard_name
                total_size += tensor.element_size() * tensor.numel()
        save_file(tensors, str(output_path / shard_name), metadata=metadata)

    updated_config = update_config(source_config, new_kv_heads=new_kv_heads)
    write_json(output_path / "config.json", updated_config)

    if source_args is not None:
        updated_args = update_args(source_args, new_kv_heads=new_kv_heads)
        write_json(output_path / "args.json", updated_args)

    output_index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    write_json(output_path / "model.safetensors.index.json", output_index)

    manifest = build_manifest(
        source_model=model_path,
        output_path=output_path,
        old_kv_heads=old_kv_heads,
        new_kv_heads=new_kv_heads,
        full_attention_layers=full_attention_layers,
        modified_tensors=modified_tensors,
        preserved_mtp_keys=preserved_mtp_keys,
    )
    write_json(output_path / "prune_manifest.json", manifest)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--new-kv-heads", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    reduce_kv_heads_qwen_averaging(
        model_path=cli_args.model,
        output_path=cli_args.out,
        new_kv_heads=cli_args.new_kv_heads,
    )
