#!/usr/bin/env python3

from __future__ import annotations

import argparse
import difflib
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from layer_loss_probe import (
    ablate_decoder_layer,
    find_text_layers,
    get_input_device,
    get_layer_types,
    get_text_config,
    load_dataset,
    load_model_and_processor,
    maybe_synchronize,
    move_tensor_dict,
    normalize_image_paths,
    normalize_messages,
    open_images,
    prepare_messages_for_processor,
)


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
        "/root/data/fixed_dataset.json",
        "/root/data/val.json",
        "/root/data/train.json",
        "fixed_dataset.json",
        "val.json",
        "train.json",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return "fixed_dataset.json"


def parse_indices(raw: str) -> List[int]:
    if not raw.strip():
        return []
    return sorted({int(part.strip()) for part in raw.split(",") if part.strip()})


def spread_sample_indices(total: int, count: int) -> List[int]:
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))

    indices: List[int] = []
    for i in range(count):
        start = (i * total) // count
        end = ((i + 1) * total) // count
        if end <= start:
            idx = start
        else:
            idx = start + ((end - start) - 1) // 2
        indices.append(idx)
    return indices


def select_examples(
    dataset: Sequence[Dict[str, Any]],
    *,
    num_examples: int,
    sample_mode: str,
    seed: int,
    sample_indices: Sequence[int],
) -> List[int]:
    if sample_indices:
        return list(sample_indices)
    if num_examples <= 0:
        raise ValueError("--num-examples must be > 0")
    if not dataset:
        raise ValueError("Dataset is empty")

    count = min(num_examples, len(dataset))
    if sample_mode == "first":
        return list(range(count))
    if sample_mode == "spread":
        return spread_sample_indices(len(dataset), count)

    rng = random.Random(seed)
    return sorted(rng.sample(range(len(dataset)), k=count))


def maybe_move_for_cache(
    data: Dict[str, torch.Tensor],
    cache_device: str,
    input_device: torch.device,
) -> Dict[str, torch.Tensor]:
    if cache_device == "input":
        return move_tensor_dict(data, input_device)
    return move_tensor_dict(data, torch.device("cpu"))


def norm_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = " ".join(cleaned.split())
    return cleaned


def similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()


def short_excerpt(text: str, limit: int = 300) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n... [truncated]"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def decode_tokens(processor: Any, token_ids: torch.Tensor) -> str:
    ids = token_ids.tolist()
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return processor.decode(ids, skip_special_tokens=True)


@dataclass
class PreparedGenerationExample:
    dataset_idx: int
    prompt_inputs: Dict[str, torch.Tensor]
    prompt_length: int
    target_ids: torch.Tensor
    prompt_text: str
    reference_text: str
    target_prefix_text: str
    image_paths: List[str]


@dataclass
class ExampleProbeResult:
    dataset_idx: int
    generated_tokens: int
    target_prefix_tokens: int
    prefix_token_acc: float
    prefix_exact: bool
    text_similarity: float
    generated_text: str
    target_prefix_text: str
    image_paths: List[str]


@dataclass
class ProbeSummary:
    avg_prefix_token_acc: float
    avg_text_similarity: float
    prefix_exact_count: int
    avg_generated_tokens: float
    elapsed_seconds: float
    examples: List[ExampleProbeResult]


@dataclass
class LayerProbeResult:
    layer_idx: int
    layer_type: str
    avg_prefix_token_acc: float
    avg_text_similarity: float
    prefix_exact_count: int
    avg_generated_tokens: float
    delta_prefix_token_acc: float
    delta_text_similarity: float
    elapsed_seconds: float
    examples: List[ExampleProbeResult]


def build_generation_inputs(
    processor: Any,
    messages: List[Dict[str, Any]],
    image_paths: Sequence[Path],
    max_new_tokens: int,
) -> tuple[Dict[str, torch.Tensor], int, torch.Tensor, str, str, str]:
    messages = prepare_messages_for_processor(processor, messages)

    if len(messages) < 2:
        raise ValueError("Need at least a user turn and an assistant turn")
    if messages[-1].get("role") != "assistant":
        raise ValueError("Final message must be the assistant target")

    prompt_messages = messages[:-1]
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    images = open_images(image_paths) if image_paths else None
    if images:
        prompt_inputs = processor(text=prompt_text, images=images, return_tensors="pt")
        full_inputs = processor(text=full_text, images=images, return_tensors="pt")
    else:
        prompt_inputs = processor(text=prompt_text, return_tensors="pt")
        full_inputs = processor(text=full_text, return_tensors="pt")

    prompt_len = int(prompt_inputs["input_ids"].shape[-1])
    full_input_ids = full_inputs["input_ids"][0]
    if "attention_mask" in full_inputs:
        full_input_ids = full_input_ids[full_inputs["attention_mask"][0].bool()]
    if prompt_len >= full_input_ids.shape[0]:
        raise ValueError("Prompt consumed the entire sequence; no assistant tokens remain")

    target_ids = full_input_ids[prompt_len:].clone().cpu()
    target_prefix_ids = target_ids[:max_new_tokens].clone()
    target_prefix_text = decode_tokens(processor, target_prefix_ids)
    reference_text = messages[-1].get("content", "")

    return prompt_inputs, prompt_len, target_ids, prompt_text, reference_text, target_prefix_text


def prepare_examples(
    processor: Any,
    dataset_path: Path,
    dataset: Sequence[Dict[str, Any]],
    indices: Sequence[int],
    *,
    max_new_tokens: int,
    cache_device: str,
    input_device: torch.device,
) -> List[PreparedGenerationExample]:
    prepared: List[PreparedGenerationExample] = []
    for dataset_idx in indices:
        example = dataset[dataset_idx]
        messages = normalize_messages(example)
        image_paths = normalize_image_paths(example, dataset_path)
        (
            prompt_inputs,
            prompt_len,
            target_ids,
            prompt_text,
            reference_text,
            target_prefix_text,
        ) = build_generation_inputs(processor, messages, image_paths, max_new_tokens)

        prepared.append(
            PreparedGenerationExample(
                dataset_idx=dataset_idx,
                prompt_inputs=maybe_move_for_cache(prompt_inputs, cache_device, input_device),
                prompt_length=prompt_len,
                target_ids=target_ids,
                prompt_text=prompt_text,
                reference_text=reference_text,
                target_prefix_text=target_prefix_text,
                image_paths=[str(path) for path in image_paths],
            )
        )
    return prepared


def run_generation_probe(
    model: torch.nn.Module,
    processor: Any,
    examples: Sequence[PreparedGenerationExample],
    *,
    input_device: torch.device,
    max_new_tokens: int,
) -> ProbeSummary:
    tokenizer = getattr(processor, "tokenizer", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    results: List[ExampleProbeResult] = []
    maybe_synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for example in examples:
            batch = move_tensor_dict(example.prompt_inputs, input_device)
            generated = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                num_beams=1,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
            generated_ids = generated[0, example.prompt_length :].detach().cpu()

            target_prefix_len = min(max_new_tokens, int(example.target_ids.shape[0]))
            target_prefix_ids = example.target_ids[:target_prefix_len]
            overlap = min(int(generated_ids.shape[0]), target_prefix_len)
            matches = 0
            if overlap > 0:
                matches = int((generated_ids[:overlap] == target_prefix_ids[:overlap]).sum().item())
            prefix_token_acc = (matches / target_prefix_len) if target_prefix_len > 0 else 0.0
            prefix_exact = (
                target_prefix_len > 0
                and int(generated_ids.shape[0]) >= target_prefix_len
                and bool(torch.equal(generated_ids[:target_prefix_len], target_prefix_ids))
            )
            generated_prefix_text = decode_tokens(processor, generated_ids[:max_new_tokens])
            results.append(
                ExampleProbeResult(
                    dataset_idx=example.dataset_idx,
                    generated_tokens=int(generated_ids.shape[0]),
                    target_prefix_tokens=target_prefix_len,
                    prefix_token_acc=prefix_token_acc,
                    prefix_exact=prefix_exact,
                    text_similarity=similarity(generated_prefix_text, example.target_prefix_text),
                    generated_text=generated_prefix_text,
                    target_prefix_text=example.target_prefix_text,
                    image_paths=example.image_paths,
                )
            )
    maybe_synchronize()
    elapsed = time.perf_counter() - start

    count = len(results)
    return ProbeSummary(
        avg_prefix_token_acc=(sum(row.prefix_token_acc for row in results) / count) if count else 0.0,
        avg_text_similarity=(sum(row.text_similarity for row in results) / count) if count else 0.0,
        prefix_exact_count=sum(1 for row in results if row.prefix_exact),
        avg_generated_tokens=(sum(row.generated_tokens for row in results) / count) if count else 0.0,
        elapsed_seconds=elapsed,
        examples=results,
    )


def render_report(
    *,
    model_path: Path,
    dataset_path: Path,
    example_indices: Sequence[int],
    max_new_tokens: int,
    layer_types: Sequence[str],
    baseline: ProbeSummary,
    results: Sequence[LayerProbeResult],
    layer_type_filter: str,
    include_endpoints: bool,
    full_attention_interval: Optional[int],
) -> str:
    utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: List[str] = []
    lines.append("# Layer Inference Probe")
    lines.append("")
    lines.append(f"- Generated: `{utc_now}`")
    lines.append(f"- Model: `{model_path}`")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append(f"- Example indices: `{list(example_indices)}`")
    lines.append(f"- Max new tokens: `{max_new_tokens}`")
    lines.append(f"- Layer filter: `{layer_type_filter}`")
    lines.append(f"- Include endpoints: `{include_endpoints}`")
    if full_attention_interval:
        lines.append(f"- Full-attention interval: `{full_attention_interval}`")
    lines.append("")
    lines.append("## Baseline")
    lines.append("")
    lines.append(f"- Avg prefix token accuracy: `{baseline.avg_prefix_token_acc:.4f}`")
    lines.append(f"- Avg text similarity: `{baseline.avg_text_similarity:.4f}`")
    lines.append(f"- Prefix exact matches: `{baseline.prefix_exact_count}/{len(baseline.examples)}`")
    lines.append(f"- Avg generated tokens: `{baseline.avg_generated_tokens:.1f}`")
    lines.append(f"- Elapsed seconds: `{baseline.elapsed_seconds:.1f}`")
    lines.append("")
    lines.append("## Baseline Per Example")
    lines.append("")
    baseline_rows = [
        [
            row.dataset_idx,
            f"{row.prefix_token_acc:.4f}",
            str(row.prefix_exact).lower(),
            f"{row.text_similarity:.4f}",
            row.generated_tokens,
        ]
        for row in baseline.examples
    ]
    lines.extend(
        markdown_table(
            ["dataset_idx", "prefix_token_acc", "prefix_exact", "text_similarity", "generated_tokens"],
            baseline_rows,
        )
    )
    lines.append("")

    lines.append("## Evaluation Order")
    lines.append("")
    eval_rows = [
        [
            result.layer_idx,
            result.layer_type,
            f"{result.avg_prefix_token_acc:.4f}",
            f"{result.delta_prefix_token_acc:+.4f}",
            f"{result.avg_text_similarity:.4f}",
            result.prefix_exact_count,
            f"{result.avg_generated_tokens:.1f}",
            f"{result.elapsed_seconds:.1f}",
        ]
        for result in results
    ]
    lines.extend(
        markdown_table(
            [
                "layer",
                "type",
                "avg_prefix_tok_acc",
                "delta_acc",
                "avg_text_sim",
                "prefix_exact",
                "avg_gen_tok",
                "elapsed_s",
            ],
            eval_rows,
        )
    )
    lines.append("")

    if results:
        safest = sorted(
            results,
            key=lambda row: (
                -row.delta_prefix_token_acc,
                -row.avg_text_similarity,
                -row.prefix_exact_count,
                row.layer_idx,
            ),
        )
        lines.append("## Safest To Drop")
        lines.append("")
        safe_rows = [
            [
                rank,
                result.layer_idx,
                result.layer_type,
                f"{result.avg_prefix_token_acc:.4f}",
                f"{result.delta_prefix_token_acc:+.4f}",
                f"{result.avg_text_similarity:.4f}",
                result.prefix_exact_count,
            ]
            for rank, result in enumerate(safest, start=1)
        ]
        lines.extend(
            markdown_table(
                [
                    "rank",
                    "layer",
                    "type",
                    "avg_prefix_tok_acc",
                    "delta_acc",
                    "avg_text_sim",
                    "prefix_exact",
                ],
                safe_rows,
            )
        )
        lines.append("")

        harmful = list(reversed(safest))
        lines.append("## Most Harmful")
        lines.append("")
        harmful_rows = [
            [
                rank,
                result.layer_idx,
                result.layer_type,
                f"{result.avg_prefix_token_acc:.4f}",
                f"{result.delta_prefix_token_acc:+.4f}",
                f"{result.avg_text_similarity:.4f}",
                result.prefix_exact_count,
            ]
            for rank, result in enumerate(harmful, start=1)
        ]
        lines.extend(
            markdown_table(
                [
                    "rank",
                    "layer",
                    "type",
                    "avg_prefix_tok_acc",
                    "delta_acc",
                    "avg_text_sim",
                    "prefix_exact",
                ],
                harmful_rows,
            )
        )
        lines.append("")

    detail_layers = []
    if results:
        safe_sorted = sorted(results, key=lambda row: (-row.delta_prefix_token_acc, row.layer_idx))
        harmful_sorted = sorted(results, key=lambda row: (row.delta_prefix_token_acc, row.layer_idx))
        seen = set()
        for row in safe_sorted[:3] + harmful_sorted[:3]:
            if row.layer_idx in seen:
                continue
            detail_layers.append(row)
            seen.add(row.layer_idx)

    for result in detail_layers:
        lines.append(f"## Layer {result.layer_idx}")
        lines.append("")
        lines.append(f"- Type: `{result.layer_type}`")
        lines.append(f"- Avg prefix token accuracy: `{result.avg_prefix_token_acc:.4f}`")
        lines.append(f"- Delta vs baseline: `{result.delta_prefix_token_acc:+.4f}`")
        lines.append(f"- Avg text similarity: `{result.avg_text_similarity:.4f}`")
        lines.append(f"- Prefix exact matches: `{result.prefix_exact_count}/{len(result.examples)}`")
        lines.append(f"- Avg generated tokens: `{result.avg_generated_tokens:.1f}`")
        lines.append(f"- Elapsed seconds: `{result.elapsed_seconds:.1f}`")
        lines.append("")
        rows = [
            [
                ex.dataset_idx,
                f"{ex.prefix_token_acc:.4f}",
                str(ex.prefix_exact).lower(),
                f"{ex.text_similarity:.4f}",
                ex.generated_tokens,
            ]
            for ex in result.examples
        ]
        lines.extend(
            markdown_table(
                ["dataset_idx", "prefix_token_acc", "prefix_exact", "text_similarity", "generated_tokens"],
                rows,
            )
        )
        lines.append("")
        if result.examples:
            example = result.examples[0]
            lines.append(f"Example fixed idx `{example.dataset_idx}` target prefix:")
            lines.append("```html")
            lines.append(short_excerpt(example.target_prefix_text))
            lines.append("```")
            lines.append("")
            lines.append(f"Example fixed idx `{example.dataset_idx}` generated prefix:")
            lines.append("```html")
            lines.append(short_excerpt(example.generated_text))
            lines.append("```")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure 64-token greedy generation quality for in-memory single-layer ablations "
            "against ground-truth assistant prefixes."
        )
    )
    parser.add_argument("--model", type=str, default=default_model_path(), help="HF-format checkpoint path.")
    parser.add_argument("--dataset", type=str, default=default_dataset_path(), help="JSON or JSONL dataset.")
    parser.add_argument("--output-md", type=str, default="layer_inference_probe.md", help="Markdown report path.")
    parser.add_argument("--output-json", type=str, default="", help="Optional JSON output path.")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of examples to evaluate.")
    parser.add_argument(
        "--sample-mode",
        choices=["first", "random", "spread"],
        default="spread",
        help="How to choose the scored examples when --sample-indices is unset.",
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default="",
        help="Optional comma-separated dataset indices. If set, --num-examples is ignored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for random sampling.")
    parser.add_argument("--device-map", type=str, default="auto", help="Transformers device_map value.")
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
        help="Only evaluate layers of this type after endpoint filtering.",
    )
    parser.add_argument(
        "--layer-indices",
        type=str,
        default="",
        help="Optional comma-separated layer indices to probe.",
    )
    parser.add_argument(
        "--include-endpoints",
        action="store_true",
        help="Include the first and last text layers in the sweep.",
    )
    parser.add_argument(
        "--cache-device",
        choices=["input", "cpu"],
        default="input",
        help="Where to keep prompt tensors between ablation runs.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Greedy generation length per example.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()
    output_md_path = Path(args.output_md).expanduser().resolve()
    output_json_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else output_md_path.with_suffix(".json")
    )

    model, processor = load_model_and_processor(args)
    input_device = get_input_device(model)

    dataset = load_dataset(dataset_path)
    sample_indices = parse_indices(args.sample_indices)
    example_indices = select_examples(
        dataset=dataset,
        num_examples=args.num_examples,
        sample_mode=args.sample_mode,
        seed=args.seed,
        sample_indices=sample_indices,
    )

    prepared_examples = prepare_examples(
        processor=processor,
        dataset_path=dataset_path,
        dataset=dataset,
        indices=example_indices,
        max_new_tokens=args.max_new_tokens,
        cache_device=args.cache_device,
        input_device=input_device,
    )

    layers = find_text_layers(model)
    layer_types = get_layer_types(model, len(layers))
    text_config = get_text_config(model)
    full_attention_interval = getattr(text_config, "full_attention_interval", None)

    baseline = run_generation_probe(
        model,
        processor,
        prepared_examples,
        input_device=input_device,
        max_new_tokens=args.max_new_tokens,
    )

    results: List[LayerProbeResult] = []
    report_text = render_report(
        model_path=model_path,
        dataset_path=dataset_path,
        example_indices=example_indices,
        max_new_tokens=args.max_new_tokens,
        layer_types=layer_types,
        baseline=baseline,
        results=results,
        layer_type_filter=args.layer_type_filter,
        include_endpoints=args.include_endpoints,
        full_attention_interval=full_attention_interval,
    )
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(report_text, encoding="utf-8")
    output_json_path.write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "dataset_path": str(dataset_path),
                "example_indices": example_indices,
                "max_new_tokens": args.max_new_tokens,
                "baseline": asdict(baseline),
                "results": [asdict(result) for result in results],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    candidate_indices = list(range(len(layers)))
    if not args.include_endpoints and len(candidate_indices) > 2:
        candidate_indices = candidate_indices[1:-1]
    if args.layer_type_filter != "all":
        candidate_indices = [idx for idx in candidate_indices if layer_types[idx] == args.layer_type_filter]
    requested_layer_indices = parse_indices(args.layer_indices)
    if requested_layer_indices:
        requested = set(requested_layer_indices)
        candidate_indices = [idx for idx in candidate_indices if idx in requested]

    print(
        f"Baseline avg_prefix_tok_acc={baseline.avg_prefix_token_acc:.4f} "
        f"avg_text_sim={baseline.avg_text_similarity:.4f} "
        f"prefix_exact={baseline.prefix_exact_count}/{len(baseline.examples)} "
        f"time={baseline.elapsed_seconds:.1f}s"
    )
    print(f"Evaluating {len(candidate_indices)} ablations...")

    for layer_idx in candidate_indices:
        maybe_synchronize()
        start = time.perf_counter()
        with ablate_decoder_layer(layers[layer_idx]):
            ablated = run_generation_probe(
                model,
                processor,
                prepared_examples,
                input_device=input_device,
                max_new_tokens=args.max_new_tokens,
            )
        maybe_synchronize()
        elapsed = time.perf_counter() - start

        result = LayerProbeResult(
            layer_idx=layer_idx,
            layer_type=layer_types[layer_idx],
            avg_prefix_token_acc=ablated.avg_prefix_token_acc,
            avg_text_similarity=ablated.avg_text_similarity,
            prefix_exact_count=ablated.prefix_exact_count,
            avg_generated_tokens=ablated.avg_generated_tokens,
            delta_prefix_token_acc=ablated.avg_prefix_token_acc - baseline.avg_prefix_token_acc,
            delta_text_similarity=ablated.avg_text_similarity - baseline.avg_text_similarity,
            elapsed_seconds=elapsed,
            examples=ablated.examples,
        )
        results.append(result)

        report_text = render_report(
            model_path=model_path,
            dataset_path=dataset_path,
            example_indices=example_indices,
            max_new_tokens=args.max_new_tokens,
            layer_types=layer_types,
            baseline=baseline,
            results=results,
            layer_type_filter=args.layer_type_filter,
            include_endpoints=args.include_endpoints,
            full_attention_interval=full_attention_interval,
        )
        output_md_path.write_text(report_text, encoding="utf-8")
        output_json_path.write_text(
            json.dumps(
                {
                    "model_path": str(model_path),
                    "dataset_path": str(dataset_path),
                    "example_indices": example_indices,
                    "max_new_tokens": args.max_new_tokens,
                    "baseline": asdict(baseline),
                    "results": [asdict(item) for item in results],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            f"layer={layer_idx:02d} type={layer_types[layer_idx]:>16} "
            f"avg_prefix_tok_acc={result.avg_prefix_token_acc:.4f} "
            f"delta={result.delta_prefix_token_acc:+.4f} "
            f"avg_text_sim={result.avg_text_similarity:.4f} "
            f"prefix_exact={result.prefix_exact_count}/{len(result.examples)} "
            f"time={elapsed:.1f}s"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Finished. Report written to {output_md_path}")


if __name__ == "__main__":
    main()
