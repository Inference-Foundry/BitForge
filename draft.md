# BitForge

> *Forge smaller, run faster, measure honestly.*

**Planned** initiative exploring the theory and practice of **model quantization**—how LLMs and related neural networks are compressed from floating-point weights to lower-bit representations, and what that trade-off actually costs in quality, throughput, and memory.

## Status

Repository **TBD**—this page is the org-level placeholder until a public repo is created.

## Background

Modern foundation models are trained in FP32 or BF16, but running them locally at those precisions is often impractical. Quantization is the practice of rounding weights (and sometimes activations) to narrower types—INT8, INT4, or even INT2—to shrink the model and speed up inference. The catch is that accuracy degrades; how gracefully depends on the method, the architecture, and how carefully the compression is tuned.

BitForge is where Inference Foundry does that measurement openly.

## Intended scope

### Theory
- Taxonomy of quantization families: post-training quantization (PTQ), quantization-aware training (QAT), and hybrid approaches
- How key methods work under the hood: GPTQ, AWQ (activation-aware weight quantization), GGUF/GGML (llama.cpp formats), and bitsandbytes
- Quantization granularity: per-tensor, per-channel, per-group, and block-wise schemes
- Activation quantization vs. weight-only quantization and why the distinction matters
- Calibration datasets: how the choice of calibration data affects final quality

### Experiments
- Reproducible perplexity benchmarks across bit-widths (FP16 → INT8 → INT4 → INT2) for reference models
- Latency and memory-footprint measurements on consumer hardware (CPU and GPU)
- Side-by-side comparisons of GPTQ, AWQ, and GGUF for the same base model
- QLoRA: fine-tuning in 4-bit and measuring the rounding error introduced during adapter merge
- Outlier detection and how suppression strategies (e.g. SmoothQuant, LLM.int8()) handle activation spikes

### Integration
- Cross-links to [super-ollama](super-ollama.md) where quantized artifacts are consumed at inference time
- Guidance on picking the right format for a given runtime (llama.cpp, vLLM, HuggingFace Transformers)

## Key open questions

1. At what bit-width does a given model class start to degrade measurably on reasoning vs. factual recall tasks?
2. Does calibration data distribution matter more for smaller or larger models?
3. Can a lightweight automated benchmark detect quantization regressions fast enough to fit in a CI loop?

## How to help

When the repository exists, use its issues and contributing guide. Until then, coordinate with maintainers (see [members](https://github.com/Inference-Foundry/.github/blob/main/docs/members/README.md); roster in [`.github-private`](https://github.com/Inference-Foundry/.github-private)) or open an issue in [Inference-Foundry/.github](https://github.com/Inference-Foundry/.github/issues).

Useful backgrounds: information theory, linear algebra, familiarity with PyTorch or llama.cpp, and ideally some experience running models on real hardware.