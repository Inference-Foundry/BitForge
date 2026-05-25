import click
import json
import logging
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bitforge.calibration.dataset import CalibrationDataLoader
from bitforge.core.quantizers import QuantizationConfig, WeightQuantizer
from bitforge.evaluation.metrics import latency, memory_footprint, perplexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("bitforge")

@click.group()
def main():
    """BitForge CLI: Forge smaller, run faster, measure honestly."""
    pass

@main.command()
@click.option("--model-id", required=True, help="Hugging Face Model ID (e.g. meta-llama/Llama-2-7b-hf)")
@click.option("--bits", default=4, type=click.Choice([2, 4, 8]), help="Target quantization bit width")
@click.option("--method", default="gptq", type=click.Choice(["ptq", "gptq", "awq", "bitsandbytes"]), help="Quantization method")
@click.option("--output-path", required=True, help="Path to save the quantized model")
def quantize(model_id, bits, method, output_path):
    """Quantize an LLM using GPTQ, AWQ, or basic PTQ/bitsandbytes."""
    bits = int(bits)
    logger.info(f"Starting quantization for model '{model_id}' to {bits}-bit using method '{method}'...")

    if method != "ptq":
        raise click.ClickException(
            f"Method '{method}' is not implemented yet. Use '--method ptq' for the current runnable path."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    config = QuantizationConfig(bits=bits)
    quantizer = WeightQuantizer(config)

    stats = quantizer.quantize_model_weights(model)
    if stats["quantized_module_count"] == 0:
        raise click.ClickException("No torch.nn.Linear weights were found to quantize.")

    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    metadata = {
        "source_model": model_id,
        "method": method,
        "bits": bits,
        "format": "dequantized-ptq-hf",
        "note": "Weights were quantized and dequantized back into a standard HF checkpoint for runnable PTQ evaluation.",
        **stats,
    }
    metadata_path = os.path.join(output_path, "bitforge_quantization.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Successfully quantized model and saved to {output_path}")
    click.echo(json.dumps(metadata, indent=2))

@main.command()
@click.option("--model-id", required=True, help="Local path or Hugging Face Model ID of the quantized model")
@click.option("--dataset", default="wikitext2", help="Dataset to measure perplexity on (wikitext2, c4, etc.)")
@click.option("--device", default="cuda", help="Target execution device (cuda, cpu, mps)")
@click.option("--n-samples", default=8, show_default=True, help="Number of fixed-length eval blocks to sample")
@click.option("--seq-len", default=512, show_default=True, help="Token length for each eval block")
@click.option("--max-new-tokens", default=20, show_default=True, help="Tokens to generate for latency measurement")
def evaluate(model_id, dataset, device, n_samples, seq_len, max_new_tokens):
    """Evaluate a model's performance (perplexity, throughput, memory)."""
    logger.info(f"Evaluating model '{model_id}' on dataset '{dataset}' using device '{device}'...")

    resolved_device = _resolve_cli_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(resolved_device)

    eval_blocks = _load_eval_blocks(
        dataset_name=dataset,
        tokenizer=tokenizer,
        n_samples=n_samples,
        seq_len=seq_len,
    )
    if not eval_blocks:
        raise click.ClickException(f"No evaluation samples were loaded for dataset '{dataset}'.")

    ppl = perplexity(model, eval_blocks, device=resolved_device)
    memory = memory_footprint(model)
    latency_input = eval_blocks[0]["input_ids"]
    latency_result = latency(
        model,
        latency_input,
        num_tokens_to_generate=max_new_tokens,
        device=resolved_device,
    )

    results = {
        "model_id": model_id,
        "dataset": dataset,
        "device": resolved_device,
        "n_samples": len(eval_blocks),
        "seq_len": seq_len,
        "perplexity": ppl,
        "latency": latency_result,
        "memory": memory,
    }
    logger.info(f"Evaluation results: {results}")
    click.echo(json.dumps(results, indent=2))


def _resolve_cli_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    mps_backend = getattr(torch.backends, "mps", None)
    if device == "mps" and (mps_backend is None or not mps_backend.is_available()):
        logger.warning("MPS requested but unavailable; falling back to CPU.")
        return "cpu"
    return device


def _load_eval_blocks(dataset_name: str, tokenizer, n_samples: int, seq_len: int):
    loader = CalibrationDataLoader(tokenizer=tokenizer, seq_len=seq_len)
    dataset_key = dataset_name.lower()

    if dataset_key in {"wikitext", "wikitext2", "wiki"}:
        return loader.get_wikitext2(n_samples=n_samples, split="test")
    if dataset_key == "c4":
        return loader.get_c4(n_samples=n_samples, split="validation")
    return loader.get_custom_dataset(dataset_name, n_samples=n_samples, split="test")

if __name__ == "__main__":
    main()
