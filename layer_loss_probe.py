#!/usr/bin/env python3

import argparse
import json
import random
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForMultimodalLM,
    AutoProcessor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure assistant-token loss deltas for in-memory single-layer ablations "
            "without writing pruned checkpoints."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model_path(),
        help="HF-format checkpoint path.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset_path(),
        help="JSON or JSONL dataset. Defaults to a local val set if found.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="layer_loss_probe.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of examples to score.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["first", "random"],
        default="first",
        help="How to choose the scored examples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random example sampling.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help=(
            "Transformers device_map value. Use 'auto' for large checkpoints. "
            "Set to 'none' to load normally without accelerate device mapping."
        ),
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model load dtype.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Optional attn_implementation passed to from_pretrained.",
    )
    parser.add_argument(
        "--layer-type-filter",
        choices=["all", "linear_attention", "full_attention"],
        default="all",
        help="Only evaluate layers of this type after endpoint skipping.",
    )
    parser.add_argument(
        "--include-endpoints",
        action="store_true",
        help="Include the first and last text layers in the ablation sweep.",
    )
    parser.add_argument(
        "--cache-device",
        choices=["input", "cpu"],
        default="input",
        help=(
            "Where to keep preprocessed model inputs between ablation runs. "
            "'input' is faster and usually fine for 5 examples."
        ),
    )
    return parser.parse_args()


def default_model_path() -> str:
    candidates = [
        "/root/checkpoints/table_enrich_27B_with_base_input_0318",
        "checkpoints/model",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return "checkpoints/model"


def default_dataset_path() -> str:
    candidates = [
        "/root/data/val.json",
        "val.json",
        "/root/data/train.json",
        "train.json",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return "val.json"


def resolve_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in dataset file: {path}")
    return data


def select_examples(
    dataset: Sequence[Dict[str, Any]], num_examples: int, sample_mode: str, seed: int
) -> List[int]:
    if num_examples <= 0:
        raise ValueError("--num-examples must be > 0")
    if not dataset:
        raise ValueError("Dataset is empty")

    count = min(num_examples, len(dataset))
    if sample_mode == "first":
        return list(range(count))

    rng = random.Random(seed)
    return sorted(rng.sample(range(len(dataset)), k=count))


def resolve_example_path(raw_path: str, dataset_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (dataset_path.parent / path).resolve()


def normalize_messages(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "messages" in example:
        messages = example["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("example['messages'] must be a non-empty list")
        return messages

    if "prompt" in example and ("response" in example or "answer" in example):
        answer = example.get("response", example.get("answer"))
        return [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": answer},
        ]

    if "instruction" in example and ("output" in example or "response" in example):
        answer = example.get("output", example.get("response"))
        return [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": answer},
        ]

    raise ValueError(
        "Unsupported example format. Expected 'messages', or 'prompt'+'response', or 'instruction'+'output'."
    )


def normalize_image_paths(example: Dict[str, Any], dataset_path: Path) -> List[Path]:
    raw_images = None
    for key in ("images", "image_paths", "image", "image_path"):
        if key in example:
            raw_images = example[key]
            break

    if raw_images is None:
        return []

    if isinstance(raw_images, str):
        raw_images = [raw_images]
    elif not isinstance(raw_images, list):
        raise ValueError(f"Unsupported image field type: {type(raw_images).__name__}")

    paths = [resolve_example_path(path, dataset_path) for path in raw_images if path]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
    return paths


def open_images(paths: Sequence[Path]) -> List[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


def prepare_messages_for_processor(
    processor: Any,
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    prepared = json.loads(json.dumps(messages))

    image_token = getattr(processor, "image_token", "<image>")
    video_token = getattr(processor, "video_token", "<video>")

    for message in prepared:
        content = message.get("content")
        if isinstance(content, str):
            content = content.replace("<image>", image_token)
            content = content.replace("<video>", video_token)
            message["content"] = content

    return prepared


def build_processor_inputs(
    processor: Any,
    messages: List[Dict[str, Any]],
    image_paths: Sequence[Path],
) -> Dict[str, torch.Tensor]:
    messages = prepare_messages_for_processor(processor, messages)

    if len(messages) < 2:
        raise ValueError("Need at least a user turn and an assistant turn")
    if messages[-1].get("role") != "assistant":
        raise ValueError("Final message must be the assistant target to score")

    prompt_messages = messages[:-1]
    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    images = open_images(image_paths) if image_paths else None

    if images:
        full_inputs = processor(text=full_text, images=images, return_tensors="pt")
        prompt_inputs = processor(text=prompt_text, images=images, return_tensors="pt")
    else:
        full_inputs = processor(text=full_text, return_tensors="pt")
        prompt_inputs = processor(text=prompt_text, return_tensors="pt")

    labels = full_inputs["input_ids"].clone()
    prompt_len = int(prompt_inputs["input_ids"].shape[-1])
    if prompt_len >= labels.shape[-1]:
        raise ValueError(
            "Prompt consumed the entire sequence; no assistant target tokens remain"
        )
    labels[:, :prompt_len] = -100
    if "attention_mask" in full_inputs:
        labels = labels.masked_fill(full_inputs["attention_mask"] == 0, -100)

    full_inputs["labels"] = labels
    return full_inputs


@dataclass
class PreparedExample:
    dataset_idx: int
    target_tokens: int
    sequence_length: int
    model_inputs: Dict[str, torch.Tensor]


def get_input_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "hf_device_map"):
        for value in model.hf_device_map.values():
            if isinstance(value, int):
                return torch.device(f"cuda:{value}")
            if isinstance(value, str) and value not in {"cpu", "disk"}:
                return torch.device(value)
    return next(model.parameters()).device


def move_tensor_dict(data: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved: Dict[str, torch.Tensor] = {}
    for key, value in data.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def maybe_move_for_cache(
    data: Dict[str, torch.Tensor], cache_device: str, input_device: torch.device
) -> Dict[str, torch.Tensor]:
    if cache_device == "input":
        return data
    return move_tensor_dict(data, torch.device("cpu"))


def maybe_precompute_inputs_embeds(
    model: torch.nn.Module,
    raw_inputs: Dict[str, torch.Tensor],
    input_device: torch.device,
    cache_device: str,
) -> Dict[str, torch.Tensor]:
    on_device = move_tensor_dict(raw_inputs, input_device)

    core_model = getattr(model, "model", None)
    has_vision_inputs = "pixel_values" in on_device or "pixel_values_videos" in on_device
    can_cache_multimodal = (
        core_model is not None
        and hasattr(core_model, "get_input_embeddings")
        and hasattr(core_model, "compute_3d_position_ids")
    )

    if not has_vision_inputs or not can_cache_multimodal:
        return maybe_move_for_cache(on_device, cache_device, input_device)

    try:
        with torch.inference_mode():
            input_ids = on_device["input_ids"]
            inputs_embeds = core_model.get_input_embeddings()(input_ids)

            pixel_values = on_device.get("pixel_values")
            image_grid_thw = on_device.get("image_grid_thw")
            if pixel_values is not None:
                image_outputs = core_model.get_image_features(
                    pixel_values, image_grid_thw, return_dict=True
                )
                image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                image_mask, _ = core_model.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            pixel_values_videos = on_device.get("pixel_values_videos")
            video_grid_thw = on_device.get("video_grid_thw")
            if pixel_values_videos is not None and hasattr(core_model, "get_video_features"):
                video_outputs = core_model.get_video_features(
                    pixel_values_videos, video_grid_thw, return_dict=True
                )
                video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                _, video_mask = core_model.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            position_ids = core_model.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=on_device.get("attention_mask"),
                past_key_values=None,
                mm_token_type_ids=on_device.get("mm_token_type_ids"),
            )
    except Exception as exc:
        print(
            "warning: multimodal input precompute failed, falling back to raw inputs "
            f"for this run: {exc}"
        )
        return maybe_move_for_cache(on_device, cache_device, input_device)

    cached = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": on_device.get("attention_mask"),
        "position_ids": position_ids,
    }
    return maybe_move_for_cache(cached, cache_device, input_device)


def prepare_examples(
    processor: Any,
    dataset_path: Path,
    dataset: Sequence[Dict[str, Any]],
    indices: Sequence[int],
    model: torch.nn.Module,
    input_device: torch.device,
    cache_device: str,
) -> List[PreparedExample]:
    prepared: List[PreparedExample] = []
    for dataset_idx in indices:
        example = dataset[dataset_idx]
        messages = normalize_messages(example)
        image_paths = normalize_image_paths(example, dataset_path)
        inputs = build_processor_inputs(processor, messages, image_paths)
        target_tokens = int((inputs["labels"] != -100).sum().item())
        if target_tokens <= 0:
            raise ValueError(f"Example {dataset_idx} has no target tokens after masking")

        cached_inputs = maybe_precompute_inputs_embeds(
            model=model,
            raw_inputs=inputs,
            input_device=input_device,
            cache_device=cache_device,
        )
        cached_inputs["labels"] = (
            inputs["labels"].to(input_device if cache_device == "input" else "cpu")
        )
        prepared.append(
            PreparedExample(
                dataset_idx=dataset_idx,
                target_tokens=target_tokens,
                sequence_length=int(inputs["input_ids"].shape[-1]),
                model_inputs=cached_inputs,
            )
        )
    return prepared


def load_model_and_processor(args: argparse.Namespace) -> tuple[torch.nn.Module, Any]:
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    torch_dtype = resolve_dtype(args.torch_dtype)
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    if hasattr(config, "vision_config") and hasattr(config, "text_config"):
        model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
    else:
        try:
            model = AutoModelForMultimodalLM.from_pretrained(args.model, **model_kwargs)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    model.eval()
    return model, processor


def find_text_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    candidates = [
        ("model.language_model.layers", lambda m: m.model.language_model.layers),
        ("language_model.layers", lambda m: m.language_model.layers),
        ("model.layers", lambda m: m.model.layers),
        ("layers", lambda m: m.layers),
    ]

    for _, getter in candidates:
        try:
            layers = getter(model)
            if isinstance(layers, torch.nn.ModuleList):
                return layers
        except AttributeError:
            continue

    raise AttributeError("Could not locate text decoder layers on the loaded model")


def get_text_config(model: torch.nn.Module) -> Any:
    config = model.config
    if hasattr(config, "text_config"):
        return config.text_config
    return config


def get_layer_types(model: torch.nn.Module, num_layers: int) -> List[str]:
    text_config = get_text_config(model)
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is not None and len(layer_types) == num_layers:
        return [normalize_layer_type(x) for x in layer_types]

    if hasattr(text_config, "layers_block_type"):
        block_types = list(text_config.layers_block_type)
        if len(block_types) == num_layers:
            return [normalize_layer_type(x) for x in block_types]

    interval = getattr(text_config, "full_attention_interval", None)
    if interval and interval > 0:
        derived = []
        for layer_idx in range(num_layers):
            if (layer_idx + 1) % interval == 0:
                derived.append("full_attention")
            else:
                derived.append("linear_attention")
        return derived

    return ["unknown"] * num_layers


def normalize_layer_type(raw: str) -> str:
    if raw == "attention":
        return "full_attention"
    return raw


@dataclass
class LossSummary:
    weighted_mean_loss: float
    sample_mean_loss: float
    total_target_tokens: int
    per_example_losses: List[float]


def maybe_synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate_examples(
    model: torch.nn.Module,
    examples: Sequence[PreparedExample],
    input_device: torch.device,
) -> LossSummary:
    total_weighted_loss = 0.0
    total_target_tokens = 0
    per_example_losses: List[float] = []

    with torch.inference_mode():
        for example in examples:
            batch = move_tensor_dict(example.model_inputs, input_device)
            labels = batch.pop("labels")
            outputs = model(**batch, labels=labels, use_cache=False)
            loss = float(outputs.loss.detach().float().cpu().item())
            per_example_losses.append(loss)
            total_weighted_loss += loss * example.target_tokens
            total_target_tokens += example.target_tokens
            del outputs, batch, labels

    return LossSummary(
        weighted_mean_loss=total_weighted_loss / total_target_tokens,
        sample_mean_loss=sum(per_example_losses) / len(per_example_losses),
        total_target_tokens=total_target_tokens,
        per_example_losses=per_example_losses,
    )


@contextmanager
def ablate_decoder_layer(layer: torch.nn.Module):
    original_forward = layer.forward

    def passthrough(self, hidden_states, *args, **kwargs):
        return hidden_states

    layer.forward = types.MethodType(passthrough, layer)
    try:
        yield
    finally:
        layer.forward = original_forward


@dataclass
class LayerResult:
    layer_idx: int
    layer_type: str
    ablated_loss: float
    delta_loss: float
    delta_percent: float
    elapsed_seconds: float


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def render_report(
    *,
    model_path: Path,
    dataset_path: Path,
    example_indices: Sequence[int],
    layer_types: Sequence[str],
    baseline: LossSummary,
    results: Sequence[LayerResult],
    layer_type_filter: str,
    include_endpoints: bool,
    full_attention_interval: Optional[int],
) -> str:
    utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: List[str] = []
    lines.append("# Layer Loss Delta Probe")
    lines.append("")
    lines.append(f"- Generated: `{utc_now}`")
    lines.append(f"- Model: `{model_path}`")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append(f"- Example indices: `{list(example_indices)}`")
    lines.append(f"- Baseline weighted mean loss: `{baseline.weighted_mean_loss:.6f}`")
    lines.append(f"- Baseline sample mean loss: `{baseline.sample_mean_loss:.6f}`")
    lines.append(f"- Total target tokens: `{baseline.total_target_tokens}`")
    lines.append(f"- Layer filter: `{layer_type_filter}`")
    lines.append(f"- Include endpoints: `{include_endpoints}`")
    if full_attention_interval:
        lines.append(f"- Full-attention interval: `{full_attention_interval}`")
    lines.append("")

    lines.append("## Baseline Per Example")
    lines.append("")
    baseline_rows = [
        [idx, f"{loss:.6f}"]
        for idx, loss in zip(example_indices, baseline.per_example_losses)
    ]
    lines.extend(markdown_table(["dataset_idx", "loss"], baseline_rows))
    lines.append("")

    lines.append("## Evaluation Order")
    lines.append("")
    eval_rows = [
        [
            result.layer_idx,
            result.layer_type,
            f"{result.ablated_loss:.6f}",
            f"{result.delta_loss:+.6f}",
            f"{result.delta_percent:+.3f}%",
            f"{result.elapsed_seconds:.1f}",
        ]
        for result in results
    ]
    lines.extend(
        markdown_table(
            ["layer", "type", "ablated_loss", "delta_loss", "delta_pct", "elapsed_s"],
            eval_rows,
        )
    )
    lines.append("")

    if results:
        safest = sorted(results, key=lambda row: row.delta_loss)
        lines.append("## Safest To Drop")
        lines.append("")
        safe_rows = [
            [
                rank,
                result.layer_idx,
                result.layer_type,
                f"{result.delta_loss:+.6f}",
                f"{result.ablated_loss:.6f}",
            ]
            for rank, result in enumerate(safest, start=1)
        ]
        lines.extend(
            markdown_table(
                ["rank", "layer", "type", "delta_loss", "ablated_loss"], safe_rows
            )
        )
        lines.append("")

        most_important = list(reversed(safest))
        lines.append("## Most Important")
        lines.append("")
        important_rows = [
            [
                rank,
                result.layer_idx,
                result.layer_type,
                f"{result.delta_loss:+.6f}",
                f"{result.ablated_loss:.6f}",
            ]
            for rank, result in enumerate(most_important, start=1)
        ]
        lines.extend(
            markdown_table(
                ["rank", "layer", "type", "delta_loss", "ablated_loss"], important_rows
            )
        )
        lines.append("")

        linear_results = [row for row in safest if row.layer_type == "linear_attention"]
        if linear_results:
            lines.append("## GDN / Linear Attention Only")
            lines.append("")
            linear_rows = [
                [
                    rank,
                    result.layer_idx,
                    f"{result.delta_loss:+.6f}",
                    f"{result.ablated_loss:.6f}",
                ]
                for rank, result in enumerate(linear_results, start=1)
            ]
            lines.extend(
                markdown_table(
                    ["rank", "layer", "delta_loss", "ablated_loss"], linear_rows
                )
            )
            lines.append("")

        if full_attention_interval and linear_results:
            by_layer = {row.layer_idx: row for row in results}
            block_rows = []
            selected_layers = []
            for block_start in range(0, len(layer_types), full_attention_interval):
                block_end = min(block_start + full_attention_interval - 1, len(layer_types) - 1)
                candidates = []
                for layer_idx in range(block_start, block_end + 1):
                    if layer_types[layer_idx] != "linear_attention":
                        continue
                    if layer_idx not in by_layer:
                        continue
                    candidates.append(by_layer[layer_idx])

                if not candidates:
                    continue

                winner = min(candidates, key=lambda row: row.delta_loss)
                selected_layers.append(winner.layer_idx)
                block_rows.append(
                    [
                        f"{block_start}-{block_end}",
                        ", ".join(str(row.layer_idx) for row in candidates),
                        winner.layer_idx,
                        f"{winner.delta_loss:+.6f}",
                    ]
                )

            if block_rows:
                lines.append("## Best 1 GDN Layer Per Attention Block")
                lines.append("")
                lines.append(
                    "This is the direct shortlist for the `64L -> 48L` style plan: pick the lowest-delta "
                    "linear/GDN layer inside each full-attention interval block."
                )
                lines.append("")
                lines.extend(
                    markdown_table(
                        ["block", "gdn_candidates", "pick", "delta_loss"], block_rows
                    )
                )
                lines.append("")
                lines.append(f"Suggested layer list: `{selected_layers}`")
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()

    model_path = Path(args.model).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()
    output_path = Path(args.output_md).expanduser().resolve()

    model, processor = load_model_and_processor(args)
    input_device = get_input_device(model)

    dataset = load_dataset(dataset_path)
    example_indices = select_examples(
        dataset=dataset,
        num_examples=args.num_examples,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )
    prepared_examples = prepare_examples(
        processor=processor,
        dataset_path=dataset_path,
        dataset=dataset,
        indices=example_indices,
        model=model,
        input_device=input_device,
        cache_device=args.cache_device,
    )

    layers = find_text_layers(model)
    layer_types = get_layer_types(model, len(layers))
    text_config = get_text_config(model)
    full_attention_interval = getattr(text_config, "full_attention_interval", None)

    baseline = evaluate_examples(model, prepared_examples, input_device)
    results: List[LayerResult] = []
    write_report(
        output_path,
        render_report(
            model_path=model_path,
            dataset_path=dataset_path,
            example_indices=example_indices,
            layer_types=layer_types,
            baseline=baseline,
            results=results,
            layer_type_filter=args.layer_type_filter,
            include_endpoints=args.include_endpoints,
            full_attention_interval=full_attention_interval,
        ),
    )

    candidate_indices = list(range(len(layers)))
    if not args.include_endpoints and len(candidate_indices) > 2:
        candidate_indices = candidate_indices[1:-1]

    if args.layer_type_filter != "all":
        candidate_indices = [
            idx
            for idx in candidate_indices
            if layer_types[idx] == args.layer_type_filter
        ]

    print(
        f"Baseline weighted mean loss: {baseline.weighted_mean_loss:.6f} "
        f"across {len(prepared_examples)} examples / {baseline.total_target_tokens} target tokens"
    )
    print(f"Evaluating {len(candidate_indices)} ablations...")

    for layer_idx in candidate_indices:
        maybe_synchronize()
        start = time.perf_counter()
        with ablate_decoder_layer(layers[layer_idx]):
            ablated = evaluate_examples(model, prepared_examples, input_device)
        maybe_synchronize()
        elapsed = time.perf_counter() - start

        delta = ablated.weighted_mean_loss - baseline.weighted_mean_loss
        result = LayerResult(
            layer_idx=layer_idx,
            layer_type=layer_types[layer_idx],
            ablated_loss=ablated.weighted_mean_loss,
            delta_loss=delta,
            delta_percent=(
                (delta / baseline.weighted_mean_loss) * 100.0
                if baseline.weighted_mean_loss != 0
                else 0.0
            ),
            elapsed_seconds=elapsed,
        )
        results.append(result)

        write_report(
            output_path,
            render_report(
                model_path=model_path,
                dataset_path=dataset_path,
                example_indices=example_indices,
                layer_types=layer_types,
                baseline=baseline,
                results=results,
                layer_type_filter=args.layer_type_filter,
                include_endpoints=args.include_endpoints,
                full_attention_interval=full_attention_interval,
            ),
        )
        print(
            f"layer={layer_idx:02d} type={layer_types[layer_idx]:>16} "
            f"loss={ablated.weighted_mean_loss:.6f} delta={delta:+.6f} "
            f"time={elapsed:.1f}s"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Finished. Report written to {output_path}")


if __name__ == "__main__":
    main()
