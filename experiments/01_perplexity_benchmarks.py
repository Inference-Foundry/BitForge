"""
Experiment 01: Perplexity Benchmarks

Reproducible perplexity benchmarks across bit-widths (FP16 -> INT8 -> INT4 -> INT2)
for reference models (e.g., Llama, Mistral) on test datasets (Wikitext-2, C4).
"""

import sys
import logging
from bitforge.evaluation.benchmarks import QuantizationBenchmarkSuite

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("bitforge.experiments.01")

def run():
    logger.info("Initializing Perplexity Benchmarking Experiment...")
    model_id = "mistralai/Mistral-7B-v0.1"
    datasets = ["wikitext2", "c4"]
    
    suite = QuantizationBenchmarkSuite(base_model_id=model_id, test_datasets=datasets)
    logger.info(f"Running benchmarks for {model_id} across 16-bit, 8-bit, 4-bit, and 2-bit...")
    df = suite.run_benchmark(bit_widths=[16, 8, 4, 2])
    
    print("\n--- Benchmark Experiment Results ---")
    print(df.to_string(index=False))
    print("------------------------------------\n")

if __name__ == "__main__":
    run()
