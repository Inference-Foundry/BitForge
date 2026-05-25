import math
import time
from typing import Any, Dict, Iterable

import torch
import torch.nn as nn


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    mps_backend = getattr(torch.backends, "mps", None)
    if device == "mps" and (mps_backend is None or not mps_backend.is_available()):
        return torch.device("cpu")
    return torch.device(device)


def _iter_batches(test_inputs: Any) -> Iterable[Dict[str, torch.Tensor]]:
    if isinstance(test_inputs, torch.Tensor):
        yield {"input_ids": test_inputs}
        return

    if isinstance(test_inputs, dict):
        yield test_inputs
        return

    for item in test_inputs:
        if isinstance(item, torch.Tensor):
            yield {"input_ids": item}
        elif isinstance(item, dict):
            yield item
        else:
            yield {"input_ids": torch.tensor(item, dtype=torch.long)}


def _prepare_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    prepared = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.long)
        if value.dim() == 1:
            value = value.unsqueeze(0)
        prepared[key] = value.to(device)
    return prepared


def _extract_logits(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def perplexity(model: nn.Module, test_inputs: Any, device: str = "cuda") -> float:
    """
    Calculate the cross-entropy perplexity of a model on validation data.
    PPL = exp(loss)
    """
    device_obj = _resolve_device(device)
    model.to(device_obj)
    model.eval()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for raw_batch in _iter_batches(test_inputs):
            batch = _prepare_batch(raw_batch, device_obj)
            input_ids = batch["input_ids"]
            if input_ids.shape[-1] < 2:
                continue

            model_kwargs = {"input_ids": input_ids}
            if "attention_mask" in batch:
                model_kwargs["attention_mask"] = batch["attention_mask"]

            outputs = model(**model_kwargs)
            logits = _extract_logits(outputs)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            if "attention_mask" in batch:
                shift_attention = batch["attention_mask"][:, 1:].contiguous()
                shift_labels = shift_labels.masked_fill(shift_attention == 0, -100)

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            token_count = torch.count_nonzero(shift_labels != -100).item()
            total_loss += loss.item()
            total_tokens += token_count

    if total_tokens == 0:
        raise ValueError("Perplexity requires at least one non-padding target token.")

    return float(math.exp(total_loss / total_tokens))

def memory_footprint(model: nn.Module) -> Dict[str, float]:
    """
    Calculate model size in MB, active VRAM footprint, and parameters counts.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return {
        "static_memory_mb": size_all_mb,
        "parameter_count": sum(param.nelement() for param in model.parameters()),
        "cuda_max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    }

def latency(
    model: nn.Module,
    input_ids: torch.Tensor,
    num_tokens_to_generate: int = 50,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Measure inference latency: Time-to-First-Token (TTFT) and Average Tokens-per-Second (TPS).
    """
    device_obj = _resolve_device(device)
    model.to(device_obj)
    model.eval()

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device_obj)

    def sync_if_needed():
        if device_obj.type == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        if hasattr(model, "generate"):
            generation_kwargs = {
                "max_new_tokens": num_tokens_to_generate,
                "do_sample": False,
            }
            pad_token_id = getattr(getattr(model, "config", None), "pad_token_id", None)
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id

            first_token_kwargs = dict(generation_kwargs)
            first_token_kwargs["max_new_tokens"] = 1

            sync_if_needed()
            t0 = time.perf_counter()
            model.generate(input_ids, **first_token_kwargs)
            sync_if_needed()
            t_first = time.perf_counter()
            output_ids = model.generate(input_ids, **generation_kwargs)
            sync_if_needed()
            t_end = time.perf_counter()
            generated_tokens = max(output_ids.shape[-1] - input_ids.shape[-1], 0)
        else:
            sync_if_needed()
            t0 = time.perf_counter()
            model(input_ids=input_ids)
            sync_if_needed()
            t_first = time.perf_counter()
            t_end = t_first
            generated_tokens = 0

    ttft = t_first - t0
    total_time = t_end - t0
    decode_time = max(t_end - t_first, 0.0)
    tps = generated_tokens / decode_time if decode_time > 0 else 0.0

    return {
        "time_to_first_token_sec": ttft,
        "tokens_per_second": tps,
        "total_generation_time_sec": total_time,
        "generated_tokens": generated_tokens,
        "avg_token_latency_ms": (decode_time / generated_tokens) * 1000 if generated_tokens else 0.0,
    }

def detect_outliers(layer_activations: torch.Tensor, threshold: float = 6.0) -> Dict[str, Any]:
    """
    Identify outlier features in activations (spikes exceeding standard deviations or explicit threshold).
    These spikes often disrupt simple INT8 quantization and require suppression (SmoothQuant).
    """
    abs_activations = layer_activations.abs()
    mean = abs_activations.mean()
    std = layer_activations.std()
    max_vals = abs_activations.max(dim=0)[0]
    
    # Identify channels where max value exceeds std deviation threshold
    outlier_channels = torch.where(max_vals > mean + threshold * std)[0]
    
    return {
        "outlier_channel_count": len(outlier_channels),
        "outlier_indices": outlier_channels.tolist(),
        "max_activation_value": abs_activations.max().item()
    }
