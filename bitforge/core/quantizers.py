from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn as nn

@dataclass
class QuantizationConfig:
    bits: int = 4
    group_size: int = 128
    symmetric: bool = False
    use_double_quant: bool = False

class WeightQuantizer:
    """
    Standard Base Weight Quantizer.
    Supports basic Post-Training Quantization (PTQ) scaling and rounding.
    """
    def __init__(self, config: QuantizationConfig):
        self.config = config

    def calculate_scale_and_zeropoint(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate scales and zero-points for a given weight tensor.
        """
        max_val = tensor.max(dim=-1, keepdim=True)[0]
        min_val = tensor.min(dim=-1, keepdim=True)[0]

        if self.config.symmetric:
            qmax = (1 << (self.config.bits - 1)) - 1
            absmax = torch.maximum(max_val.abs(), min_val.abs())
            scale = torch.clamp(absmax / qmax, min=1e-5)
            zero_point = torch.zeros_like(scale)
            return scale, zero_point

        # Scale for asymmetric quantization: (max - min) / (2^b - 1)
        qmax = (1 << self.config.bits) - 1
        scale = (max_val - min_val) / qmax
        scale = torch.clamp(scale, min=1e-5) # avoid zero divisions

        zero_point = torch.round(-min_val / scale)
        return scale, zero_point

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize a tensor: W_q = round(W / scale) + zero_point
        """
        scale, zero_point = self.calculate_scale_and_zeropoint(tensor)
        
        if self.config.symmetric:
            qmin = -(1 << (self.config.bits - 1))
            qmax = (1 << (self.config.bits - 1)) - 1
        else:
            qmin = 0
            qmax = (1 << self.config.bits) - 1

        quantized = torch.round(tensor / scale) + zero_point
        quantized = torch.clamp(quantized, min=qmin, max=qmax)
        
        return quantized, scale, zero_point

    def dequantize(self, quantized: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        """
        Dequantize a tensor: W_dequant = (W_q - zero_point) * scale
        """
        return (quantized - zero_point) * scale

    def quantize_dequantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply PTQ and return a dequantized tensor that remains runnable in PyTorch.

        This preserves normal model execution while introducing the rounding error
        expected from the configured bit width.
        """
        original = tensor.detach().float()
        quantized, scale, zero_point = self.quantize(original)
        dequantized = self.dequantize(quantized, scale, zero_point).to(dtype=tensor.dtype)
        error = (original - dequantized.float()).abs()

        stats = {
            "scale_count": int(scale.numel()),
            "zero_point_count": int(zero_point.numel()),
            "mean_abs_error": float(error.mean().item()),
            "max_abs_error": float(error.max().item()),
        }
        return dequantized, stats

    def quantize_model_weights(self, model: nn.Module) -> Dict[str, float]:
        """
        Quantize/dequantize all Linear layer weights in-place.

        The saved model remains a standard PyTorch/Hugging Face checkpoint, while
        metadata reports the PTQ transformation that was applied.
        """
        module_count = 0
        parameter_count = 0
        total_mean_error = 0.0
        max_error = 0.0

        for module in model.modules():
            if not isinstance(module, nn.Linear):
                continue

            dequantized, stats = self.quantize_dequantize(module.weight.data)
            module.weight.data.copy_(dequantized)
            module_count += 1
            parameter_count += module.weight.numel()
            total_mean_error += stats["mean_abs_error"]
            max_error = max(max_error, stats["max_abs_error"])

        average_mean_error = total_mean_error / module_count if module_count else 0.0
        return {
            "quantized_module_count": module_count,
            "quantized_parameter_count": parameter_count,
            "average_mean_abs_error": average_mean_error,
            "max_abs_error": max_error,
        }
