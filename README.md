# BitForge

> **Forge smaller, run faster, measure honestly.**

BitForge is a planned initiative by **Inference Foundry** exploring the theory and practice of **model quantization**—how Large Language Models (LLMs) and related neural networks are compressed from floating-point weights to lower-bit representations, and what that trade-off actually costs in quality, throughput, and memory.

---

## 🔍 Background

Modern foundation models are trained in high-precision floats (FP32 or BF16), but running them locally at those precisions is often impractical due to hardware constraints. **Quantization** is the practice of rounding weights (and sometimes activations) to narrower types—INT8, INT4, or even INT2—to shrink the model's memory footprint and speed up inference. 

The catch is that accuracy degrades; how gracefully depends on the method, the architecture, and how carefully the compression is tuned. BitForge is where we measure those trade-offs openly and systematically.

---

## 🚀 Intended Scope

### 📚 Theory
* **Taxonomy of Quantization:** Post-training quantization (PTQ), Quantization-Aware Training (QAT), and hybrid approaches.
* **Core Methods Under the Hood:** GPTQ, AWQ (Activation-aware Weight Quantization), GGUF/GGML (llama.cpp formats), and bitsandbytes.
* **Granularity Schemes:** Per-tensor, per-channel, per-group, and block-wise quantization.
* **Weight vs. Activation:** Weight-only quantization vs. activation quantization, and why outlier suppression matters.
* **Calibration Datasets:** How the choice of calibration data affects perplexity and downstream task accuracy.

### 🧪 Experiments
* **Perplexity Benchmarks:** Reproducible evaluation across bit-widths (FP16 → INT8 → INT4 → INT2) for reference models.
* **Hardware Benchmarking:** Latency and memory-footprint measurements on consumer hardware (both CPU and GPU).
* **Cross-Method Comparisons:** Side-by-side analysis of GPTQ, AWQ, and GGUF for the same base model.
* **QLoRA Rounded Errors:** Fine-tuning in 4-bit and measuring the rounding error introduced during adapter merge.
* **Outlier Suppression:** Side-by-side evaluations of suppression strategies (e.g. SmoothQuant, LLM.int8()) in handling activation spikes.

### 🔌 Integration
* **Super-Ollama:** Cross-links to [super-ollama](super-ollama.md) where quantized artifacts are consumed at inference time.
* **Runtime Guidance:** A clear decision matrix on picking the right format for a given runtime (llama.cpp, vLLM, HuggingFace Transformers).

---

## 💻 Repository Structure

* `bitforge/`: Core Python library for running quantization, calibration, and benchmarks.
* `experiments/`: Reproducible experimental scripts and research notebooks.
* `site/`: The interactive web dashboard and visualization engine.

To explore the interactive trade-off simulator, navigate to the `site/` folder and open `index.html` in your browser.

---

## ❓ Key Questions We Focus On

1. At what bit-width does a given model class start to degrade measurably on reasoning vs. factual recall tasks?
2. Does calibration data distribution matter more for smaller or larger models?
3. Can a lightweight automated benchmark detect quantization regressions fast enough to fit in a CI loop?

---

## 🤝 Contributing

We welcome contributions! Relevant background knowledge includes information theory, linear algebra, familiarity with PyTorch or `llama.cpp`, and hands-on experience running models on consumer hardware.

Please refer to the org-wide [members list](https://github.com/Inference-Foundry/.github/blob/main/docs/members/README.md) or coordinate with the maintainers in the `.github-private` repo.