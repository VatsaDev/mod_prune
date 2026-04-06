#!/usr/bin/env python3
"""Prune HF safetensor checkpoints by removing text layers in-place.

Preserves non-text tensors such as `mtp.*` while renumbering later text layers.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")


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


def rename_key(key, remove_idx_set):
    match = LAYER_RE.match(key)
    if match is None:
        return key
    layer_idx = int(match.group(1))
    if layer_idx in remove_idx_set:
        return None
    shift = sum(1 for idx in remove_idx_set if idx < layer_idx)
    if shift == 0:
        return key
    return f"model.language_model.layers.{layer_idx - shift}.{match.group(2)}"


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def update_config(config, remove_idx):
    text_config = config.get("text_config", config)
    if "layer_types" in text_config and text_config["layer_types"] is not None:
        text_config["layer_types"] = [
            value for idx, value in enumerate(text_config["layer_types"]) if idx not in remove_idx
        ]
    if "num_hidden_layers" in text_config:
        text_config["num_hidden_layers"] -= len(remove_idx)
    if "num_hidden_layers" in config and isinstance(config["num_hidden_layers"], int):
        config["num_hidden_layers"] -= len(remove_idx)
    return config


def update_args(args_payload, remove_idx):
    remove_idx = set(remove_idx)

    def _patch_target(target):
        if not isinstance(target, dict):
            return
        if "layer_types" in target and target["layer_types"] is not None:
            target["layer_types"] = [
                value for idx, value in enumerate(target["layer_types"]) if idx not in remove_idx
            ]
        if "num_layers" in target and isinstance(target["num_layers"], int):
            target["num_layers"] -= len(remove_idx)

    _patch_target(args_payload)
    _patch_target(args_payload.get("extra_args"))
    return args_payload


def build_manifest(*, source_model, output_path, remove_idx, source_layer_types, kept_tensors, dropped_tensors, mtp_keys):
    removed_text_layers = []
    if source_layer_types is not None:
        for idx in remove_idx:
            if 0 <= idx < len(source_layer_types):
                removed_text_layers.append({
                    "zero_based_idx": idx,
                    "one_based_idx": idx + 1,
                    "layer_type": source_layer_types[idx],
                })
    return {
        "source_model": str(source_model),
        "output_path": str(output_path),
        "text_remove_idx": list(remove_idx),
        "text_remove_layer_numbers": [idx + 1 for idx in remove_idx],
        "removed_text_layers": removed_text_layers,
        "remaining_text_layers": (len(source_layer_types) - len(remove_idx)) if source_layer_types is not None else None,
        "preserved_mtp_keys": mtp_keys,
        "kept_tensors": kept_tensors,
        "dropped_tensors": dropped_tensors,
    }


def copy_auxiliary_files(source_dir, output_dir, skip_names):
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


def prune_checkpoint(source_dir, output_dir, remove_idx):
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output path already exists: {output_dir}")

    source_index = load_json(source_dir / "model.safetensors.index.json")
    source_config = load_json(source_dir / "config.json")
    source_args = load_json(source_dir / "args.json") if (source_dir / "args.json").exists() else None
    source_layer_types = source_config.get("text_config", source_config).get("layer_types")
    remove_idx = sorted(set(remove_idx))

    output_dir.mkdir(parents=True, exist_ok=False)
    skip_names = {"model.safetensors.index.json", "config.json"}
    if source_args is not None:
        skip_names.add("args.json")
    copy_auxiliary_files(source_dir, output_dir, skip_names)

    shard_names = sorted(set(source_index["weight_map"].values()))
    weight_map = {}
    total_size = 0
    kept_tensors = 0
    dropped_tensors = 0
    mtp_keys = 0

    for shard_name in shard_names:
        src_shard = source_dir / shard_name
        tensors = {}
        metadata = None
        with safe_open(src_shard, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            for key in handle.keys():
                new_key = rename_key(key, set(remove_idx))
                if new_key is None:
                    dropped_tensors += 1
                    continue
                tensor = handle.get_tensor(key)
                tensors[new_key] = tensor
                weight_map[new_key] = shard_name
                kept_tensors += 1
                total_size += tensor.element_size() * tensor.numel()
                if new_key.startswith("mtp."):
                    mtp_keys += 1
        save_file(tensors, str(output_dir / shard_name), metadata=metadata)

    updated_config = update_config(source_config, remove_idx)
    write_json(output_dir / "config.json", updated_config)

    if source_args is not None:
        updated_args = update_args(source_args, remove_idx)
        write_json(output_dir / "args.json", updated_args)

    output_index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    write_json(output_dir / "model.safetensors.index.json", output_index)

    manifest = build_manifest(
        source_model=source_dir,
        output_path=output_dir,
        remove_idx=remove_idx,
        source_layer_types=source_layer_types,
        kept_tensors=kept_tensors,
        dropped_tensors=dropped_tensors,
        mtp_keys=mtp_keys,
    )
    write_json(output_dir / "prune_manifest.json", manifest)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--text-chop", default=None, type=str, help="Comma-separated zero-based text indices.")
    parser.add_argument(
        "--text-chop-one-based", default=None, type=str, help="Comma-separated one-based text layer numbers.")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.text_chop is not None and cli_args.text_chop_one_based is not None:
        raise ValueError("Use either --text-chop or --text-chop-one-based, not both.")
    if cli_args.text_chop_one_based is not None:
        remove_idx = parse_one_based_indices(cli_args.text_chop_one_based)
    else:
        remove_idx = parse_indices(cli_args.text_chop)
    prune_checkpoint(cli_args.model, cli_args.out, remove_idx)
