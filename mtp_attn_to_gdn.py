#!/usr/bin/env python3
"""Replace the MTP self-attention block with GDN tensors in a Qwen checkpoint.

This operates directly on safetensors shards so the checkpoint keeps all existing
MTP tensors and only swaps `mtp.layers.0.self_attn.*` for
`mtp.layers.0.linear_attn.*`.
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


SOURCE_LINEAR_SUFFIXES = (
    "in_proj_qkv.weight",
    "in_proj_z.weight",
    "in_proj_b.weight",
    "in_proj_a.weight",
    "conv1d.weight",
    "dt_bias",
    "A_log",
    "norm.weight",
    "out_proj.weight",
)

OLD_MTP_ATTN_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "q_norm.weight",
    "k_norm.weight",
)


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


def get_text_config(config_payload: dict):
    return config_payload.get("text_config", config_payload)


def get_last_linear_layer_index(config_payload: dict) -> int:
    text_config = get_text_config(config_payload)
    layer_types = text_config.get("layer_types")
    if not isinstance(layer_types, list):
        raise ValueError("Expected config to contain text_config.layer_types.")
    linear_layers = [idx for idx, layer_type in enumerate(layer_types) if layer_type == "linear_attention"]
    if not linear_layers:
        raise ValueError("Checkpoint has no linear_attention decoder layers to seed GDN from.")
    return linear_layers[-1]


def get_source_linear_tensors(model_path: Path, source_index: dict, source_layer_idx: int):
    source_prefix = f"model.language_model.layers.{source_layer_idx}.linear_attn."
    required_keys = [f"{source_prefix}{suffix}" for suffix in SOURCE_LINEAR_SUFFIXES]
    shard_to_keys: dict[str, list[tuple[str, str]]] = {}
    for key in required_keys:
        shard_name = source_index["weight_map"].get(key)
        if shard_name is None:
            raise KeyError(f"Missing source tensor: {key}")
        shard_to_keys.setdefault(shard_name, []).append((key, key.removeprefix(source_prefix)))

    tensors = {}
    for shard_name, entries in shard_to_keys.items():
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as handle:
            for key, suffix in entries:
                tensors[suffix] = handle.get_tensor(key).clone()

    missing = sorted(set(SOURCE_LINEAR_SUFFIXES) - set(tensors))
    if missing:
        raise KeyError(f"Failed to collect source GDN tensors: {missing}")
    return tensors


def update_config(config_payload: dict):
    text_config = get_text_config(config_payload)
    text_config["mtp_layer_type"] = "linear_attention"
    config_payload["mtp_layer_type"] = "linear_attention"
    return config_payload


def update_args(args_payload: dict):
    args_payload["mtp_layer_type"] = "linear_attention"
    extra_args = args_payload.get("extra_args")
    if isinstance(extra_args, dict):
        extra_args["mtp_layer_type"] = "linear_attention"
    return args_payload


def build_manifest(
    *,
    source_model: Path,
    output_path: Path,
    source_linear_layer_idx: int,
    removed_tensors: list[str],
    added_tensors: list[str],
):
    return {
        "source_model": str(source_model),
        "output_path": str(output_path),
        "mtp_layer_type": "linear_attention",
        "seed_decoder_layer_idx": source_linear_layer_idx,
        "seed_decoder_layer_number": source_linear_layer_idx + 1,
        "removed_tensor_count": len(removed_tensors),
        "removed_tensors": removed_tensors,
        "added_tensor_count": len(added_tensors),
        "added_tensors": added_tensors,
    }


def convert_mtp_attn_to_gdn(model_path: Path, output_path: Path, *, seed_layer_idx: int | None = None):
    model_path = model_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Output path already exists: {output_path}")

    source_index = load_json(model_path / "model.safetensors.index.json")
    source_config = load_json(model_path / "config.json")
    source_args = load_json(model_path / "args.json") if (model_path / "args.json").exists() else None

    source_layer_idx = seed_layer_idx if seed_layer_idx is not None else get_last_linear_layer_index(source_config)
    source_linear_tensors = get_source_linear_tensors(model_path, source_index, source_layer_idx)

    old_mtp_attn_keys = [f"mtp.layers.0.self_attn.{suffix}" for suffix in OLD_MTP_ATTN_SUFFIXES]
    old_shards = {source_index["weight_map"].get(key) for key in old_mtp_attn_keys}
    if None in old_shards:
        missing = [key for key in old_mtp_attn_keys if key not in source_index["weight_map"]]
        raise KeyError(f"Missing MTP attention tensors: {missing}")
    target_shard = next(iter(sorted(old_shards)))

    output_path.mkdir(parents=True, exist_ok=False)
    skip_names = {"model.safetensors.index.json", "config.json"}
    if source_args is not None:
        skip_names.add("args.json")
    copy_auxiliary_files(model_path, output_path, skip_names)

    weight_map = {}
    total_size = 0
    added_tensors = [f"mtp.layers.0.linear_attn.{suffix}" for suffix in SOURCE_LINEAR_SUFFIXES]

    for shard_name in sorted(set(source_index["weight_map"].values())):
        src_shard = model_path / shard_name
        tensors = {}
        metadata = None
        with safe_open(src_shard, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            for key in handle.keys():
                if key in old_mtp_attn_keys:
                    continue
                tensor = handle.get_tensor(key)
                tensors[key] = tensor
                weight_map[key] = shard_name
                total_size += tensor.element_size() * tensor.numel()

        if shard_name == target_shard:
            for suffix, tensor in source_linear_tensors.items():
                key = f"mtp.layers.0.linear_attn.{suffix}"
                tensor = tensor.clone()
                tensors[key] = tensor
                weight_map[key] = shard_name
                total_size += tensor.element_size() * tensor.numel()

        save_file(tensors, str(output_path / shard_name), metadata=metadata)

    write_json(output_path / "config.json", update_config(source_config))
    if source_args is not None:
        write_json(output_path / "args.json", update_args(source_args))
    write_json(
        output_path / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )
    write_json(
        output_path / "prune_manifest.json",
        build_manifest(
            source_model=model_path,
            output_path=output_path,
            source_linear_layer_idx=source_layer_idx,
            removed_tensors=old_mtp_attn_keys,
            added_tensors=added_tensors,
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed-layer-idx", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    convert_mtp_attn_to_gdn(
        model_path=cli_args.model,
        output_path=cli_args.out,
        seed_layer_idx=cli_args.seed_layer_idx,
    )
