from typing import Any, List, Optional

import pandas as pd
from bitforge.evaluation.metrics import perplexity, memory_footprint, latency


class QuantizationBenchmarkSuite:
    """
    Automated benchmark suite to run cross-bitwidth evaluations of models.
    Compiles memory, perplexity, and generation speed into structured summaries.
    """
    def __init__(self, base_model_id: str, test_datasets: List[str]):
        self.base_model_id = base_model_id
        self.test_datasets = test_datasets
        self.results = []

    def run_benchmark(
        self,
        bit_widths: List[int] = [16, 8, 4, 2],
        model: Optional[Any] = None,
        test_inputs: Optional[Any] = None,
        generation_input_ids: Optional[Any] = None,
        device: str = "cuda",
    ) -> pd.DataFrame:
        """
        Evaluate the model across various bit-widths.

        When no model/test inputs are supplied, this returns an explicitly simulated
        trend table so experiment scaffolding remains runnable without large downloads.
        """
        self.results = []
        if model is not None and test_inputs is not None:
            return self._run_real_benchmark(
                bit_widths=bit_widths,
                model=model,
                test_inputs=test_inputs,
                generation_input_ids=generation_input_ids,
                device=device,
            )

        return self._run_simulated_benchmark(bit_widths=bit_widths)

    def _run_simulated_benchmark(self, bit_widths: List[int]) -> pd.DataFrame:
        for bits in bit_widths:
            result = {
                "model_id": self.base_model_id,
                "bits": bits,
                "perplexity": 6.8 + (16 - bits) * 0.8,
                "vram_gb": (bits / 16.0) * 14.0,
                "tokens_per_sec": 12.5 * (16 / bits),
                "is_simulated": True,
                "note": "Simulated trend data; pass model and test_inputs for real metrics.",
            }
            self.results.append(result)

        return pd.DataFrame(self.results)

    def _run_real_benchmark(
        self,
        bit_widths: List[int],
        model: Any,
        test_inputs: Any,
        generation_input_ids: Optional[Any],
        device: str,
    ) -> pd.DataFrame:
        memory = memory_footprint(model)

        for bits in bit_widths:
            latency_result = {}
            if generation_input_ids is not None:
                latency_result = latency(model, generation_input_ids, device=device)

            result = {
                "model_id": self.base_model_id,
                "bits": bits,
                "perplexity": perplexity(model, test_inputs, device=device),
                "static_memory_mb": memory["static_memory_mb"],
                "tokens_per_sec": latency_result.get("tokens_per_second", 0.0),
                "is_simulated": False,
                "note": "Real evaluation metrics; quantization for this bit width is not applied by the suite yet.",
            }
            self.results.append(result)

        return pd.DataFrame(self.results)
