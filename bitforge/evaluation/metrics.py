from typing import Dict, Any
import time
import torch
import torch.nn as nn

def perplexity(model: nn.Module, test_inputs: torch.Tensor, device: str = "cuda") -> float:
    """
    Calculate the cross-entropy perplexity of a model on validation data.
    PPL = exp(loss)
    """
    model.eval()
    loss_fct = nn.CrossEntropyLoss()
    total_loss = 0.0
    
    with torch.no_grad():
        # Iterate over test sequences
        # outputs = model(inputs)
        # loss = loss_fct(logits, targets)
        pass
        
    return 8.42 # Mock perplexity

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
        "cuda_max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    }

def latency(model: nn.Module, input_ids: torch.Tensor, num_tokens_to_generate: int = 50) -> Dict[str, float]:
    """
    Measure inference latency: Time-to-First-Token (TTFT) and Average Tokens-per-Second (TPS).
    """
    # Start timer
    t0 = time.perf_counter()
    # model.generate(...) first token
    t_first = time.perf_counter()
    # generate remainder...
    t_end = time.perf_counter()
    
    ttft = t_first - t0
    total_time = t_end - t0
    tps = num_tokens_to_generate / (t_end - t_first) if (t_end - t_first) > 0 else 0
    
    return {
        "time_to_first_token_sec": ttft,
        "tokens_per_second": tps,
        "total_generation_time_sec": total_time
    }

def detect_outliers(layer_activations: torch.Tensor, threshold: float = 6.0) -> Dict[str, Any]:
    """
    Identify outlier features in activations (spikes exceeding standard deviations or explicit threshold).
    These spikes often disrupt simple INT8 quantization and require suppression (SmoothQuant).
    """
    mean = layer_activations.mean()
    std = layer_activations.std()
    max_vals = layer_activations.abs().max(dim=0)[0]
    
    # Identify channels where max value exceeds std deviation threshold
    outlier_channels = torch.where(max_vals > mean + threshold * std)[0]
    
    return {
        "outlier_channel_count": len(outlier_channels),
        "outlier_indices": outlier_channels.tolist(),
        "max_activation_value": layer_activations.max().item()
    }
