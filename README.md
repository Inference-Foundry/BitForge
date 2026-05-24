# BitForge

> **Forge smaller, run faster, measure honestly.**

BitForge is an [Inference Foundry](https://github.com/Inference-Foundry) project for exploring **LLM quantization**—compressing models to lower bit widths and measuring the trade-offs in quality, speed, and memory.

## Quick start

```bash
git clone https://github.com/Inference-Foundry/BitForge.git
cd BitForge
pip install -e .

bitforge --help
```

```bash
# Quantize
bitforge quantize --model-id meta-llama/Llama-2-7b-hf --bits 4 --method gptq --output-path ./out/model

# Evaluate
bitforge evaluate --model-id ./out/model --dataset wikitext2 --device cuda
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 2.1, GPU recommended for quantization.

## What's in this repo

| Path | Purpose |
|------|---------|
| `bitforge/` | Python library — quantizers (GPTQ, AWQ, PTQ), calibration, metrics, CLI |
| `experiments/` | Reproducible benchmark scripts (perplexity, latency, QLoRA merge, outliers) |
| `site/` | Interactive quantization dashboard — open `site/index.html` in a browser |

## Documentation

**Full documentation lives in the [GitHub Wiki](https://github.com/Inference-Foundry/BitForge/wiki).**

| Wiki page | Topics |
|-----------|--------|
| [Home](https://github.com/Inference-Foundry/BitForge/wiki) | Overview and research questions |
| [Getting Started](https://github.com/Inference-Foundry/BitForge/wiki/Getting-Started) | Install, hardware notes, first commands |
| [Architecture](https://github.com/Inference-Foundry/BitForge/wiki/Architecture) | Package layout and module map |
| [Quantization Theory](https://github.com/Inference-Foundry/BitForge/wiki/Quantization-Theory) | Methods, granularity, calibration |
| [Experiments](https://github.com/Inference-Foundry/BitForge/wiki/Experiments) | Running benchmark scripts |
| [CLI Reference](https://github.com/Inference-Foundry/BitForge/wiki/CLI-Reference) | `quantize` and `evaluate` |
| [Runtime Integration](https://github.com/Inference-Foundry/BitForge/wiki/Runtime-Integration) | llama.cpp, vLLM, Super-Ollama |
| [Interactive Lab](https://github.com/Inference-Foundry/BitForge/wiki/Interactive-Lab) | Web dashboard guide |
| [Contributing](https://github.com/Inference-Foundry/BitForge/wiki/Contributing) | How to help |

## Status

Early-stage (v0.1.0). Core interfaces and experiment scaffolding are in place; some pipelines return placeholder data until full calibration loops land. See the wiki [Architecture](https://github.com/Inference-Foundry/BitForge/wiki/Architecture) page for details.

## Related

- [Super-Ollama](https://github.com/Inference-Foundry/super-ollama) — serves quantized GGUF models at inference time
- [Inference Foundry](https://github.com/Inference-Foundry) — org hub

## Contributing

Contributions welcome. See the wiki [Contributing](https://github.com/Inference-Foundry/BitForge/wiki/Contributing) page and open an [issue](https://github.com/Inference-Foundry/BitForge/issues) before large changes.
