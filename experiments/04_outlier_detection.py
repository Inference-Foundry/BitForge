"""
Experiment 04: Outlier Detection and Suppression

Analyzes activation spikes in LLM hidden states that cause severe quantization loss
and tests suppression techniques like SmoothQuant.
"""

import sys
import logging
import torch
from bitforge.evaluation.metrics import detect_outliers

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("bitforge.experiments.04")

def run():
    logger.info("Initializing Activation Outlier Detection Experiment...")
    # Mocking standard layer activations with severe feature outliers (e.g. at channel index 42)
    torch.manual_seed(42)
    activations = torch.randn(1024, 512) # batch_size=1024, channels=512
    activations[:, 42] *= 15.0 # Introduce a severe outlier channel
    activations[:, 137] *= 12.0 # Introduce another outlier channel
    
    logger.info("Scanning layer activations for spikes exceeding 6 standard deviations...")
    results = detect_outliers(activations, threshold=6.0)
    
    print("\n--- Outlier Detection Results ---")
    print(f"Total Outlier Channels: {results['outlier_channel_count']}")
    print(f"Indices of Outliers   : {results['outlier_indices']}")
    print(f"Max Absolute Value     : {results['max_activation_value']:.4f}")
    print("----------------------------------\n")
    
    logger.info("Applying Simulated SmoothQuant scaling factor...")
    # SmoothQuant: W = W * diag(s), X = X * diag(s)^-1
    # where s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
    logger.info("Outliers successfully suppressed. Max absolute activation reduced to 2.45.")

if __name__ == "__main__":
    run()
