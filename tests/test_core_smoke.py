import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner
import torch
import torch.nn as nn

from bitforge.calibration.dataset import CalibrationDataLoader
from bitforge.cli import main
from bitforge.core.quantizers import QuantizationConfig, WeightQuantizer
from bitforge.evaluation.benchmarks import QuantizationBenchmarkSuite
from bitforge.evaluation.metrics import detect_outliers, latency, memory_footprint, perplexity


class DummyTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False, return_attention_mask=False):
        return {"input_ids": list(range(1, len(text.split()) + 1))}

    def save_pretrained(self, output_path):
        with open(os.path.join(output_path, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump({"tokenizer_class": "DummyTokenizer"}, f)


class UniformLogitModel(nn.Module):
    def __init__(self, vocab_size=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.param = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(pad_token_id=0)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=input_ids.device)
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, max_new_tokens=1, do_sample=False, **kwargs):
        next_ids = torch.ones(
            input_ids.shape[0],
            max_new_tokens,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, next_ids], dim=-1)


class SaveableLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(2, 2, bias=False)
        self.config = SimpleNamespace(pad_token_id=0)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.tensor([[0.0, 0.3], [0.6, 1.0]]))

    def forward(self, input_ids, attention_mask=None):
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 2, device=input_ids.device)
        return SimpleNamespace(logits=logits)

    def save_pretrained(self, output_path):
        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))


class WeightQuantizerTests(unittest.TestCase):
    def test_quantize_returns_expected_shapes_and_ranges(self):
        tensor = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [2.0, 3.0, 4.0],
            ]
        )
        quantizer = WeightQuantizer(QuantizationConfig(bits=4))

        quantized, scale, zero_point = quantizer.quantize(tensor)

        self.assertEqual(quantized.shape, tensor.shape)
        self.assertEqual(scale.shape, (2, 1))
        self.assertEqual(zero_point.shape, (2, 1))
        self.assertTrue(torch.all(quantized >= 0))
        self.assertTrue(torch.all(quantized <= 15))

    def test_constant_tensor_does_not_produce_invalid_scale(self):
        tensor = torch.full((2, 3), 5.0)
        quantizer = WeightQuantizer(QuantizationConfig(bits=8))

        quantized, scale, zero_point = quantizer.quantize(tensor)
        dequantized = quantizer.dequantize(quantized, scale, zero_point)

        self.assertTrue(torch.all(torch.isfinite(scale)))
        self.assertTrue(torch.all(torch.isfinite(dequantized)))

    def test_quantize_model_weights_updates_linear_weights_and_reports_stats(self):
        model = SaveableLinearModel()
        original = model.lm_head.weight.detach().clone()
        quantizer = WeightQuantizer(QuantizationConfig(bits=2))

        stats = quantizer.quantize_model_weights(model)

        self.assertEqual(stats["quantized_module_count"], 1)
        self.assertEqual(stats["quantized_parameter_count"], 4)
        self.assertFalse(torch.equal(model.lm_head.weight, original))
        self.assertGreaterEqual(stats["max_abs_error"], 0.0)


class MetricTests(unittest.TestCase):
    def test_calibration_loader_slices_text_into_fixed_blocks(self):
        loader = CalibrationDataLoader(DummyTokenizer(), seq_len=4)

        blocks = loader.get_from_texts(["a b c d e f g"], n_samples=2)

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["input_ids"].tolist(), [1, 2, 3, 4])
        self.assertEqual(blocks[0]["attention_mask"].tolist(), [1, 1, 1, 1])
        self.assertEqual(blocks[1]["input_ids"].shape, (4,))

    def test_perplexity_uses_model_logits(self):
        model = UniformLogitModel(vocab_size=5)
        inputs = [{"input_ids": torch.tensor([1, 2, 3, 4])}]

        result = perplexity(model, inputs, device="cpu")

        self.assertAlmostEqual(result, 5.0, places=4)

    def test_latency_uses_generate_when_available(self):
        model = UniformLogitModel(vocab_size=5)

        result = latency(model, torch.tensor([1, 2, 3]), num_tokens_to_generate=2, device="cpu")

        self.assertEqual(result["generated_tokens"], 2)
        self.assertIn("tokens_per_second", result)

    def test_detect_outliers_identifies_spike_channels(self):
        activations = torch.zeros((8, 4))
        activations[:, 2] = 10.0

        result = detect_outliers(activations, threshold=1.0)

        self.assertEqual(result["outlier_channel_count"], 1)
        self.assertEqual(result["outlier_indices"], [2])
        self.assertEqual(result["max_activation_value"], 10.0)

    def test_memory_footprint_reports_static_model_size(self):
        model = nn.Linear(3, 2)

        result = memory_footprint(model)

        self.assertGreater(result["static_memory_mb"], 0)
        self.assertIn("cuda_max_memory_allocated_mb", result)


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_default_results_are_marked_simulated(self):
        suite = QuantizationBenchmarkSuite("dummy-model", ["wikitext2"])

        df = suite.run_benchmark(bit_widths=[16, 4])

        self.assertEqual(len(df), 2)
        self.assertTrue(df["is_simulated"].all())

    def test_benchmark_can_run_real_metric_path(self):
        suite = QuantizationBenchmarkSuite("dummy-model", ["wikitext2"])
        model = UniformLogitModel(vocab_size=5)
        inputs = [{"input_ids": torch.tensor([1, 2, 3, 4])}]

        df = suite.run_benchmark(
            bit_widths=[16],
            model=model,
            test_inputs=inputs,
            generation_input_ids=torch.tensor([1, 2, 3]),
            device="cpu",
        )

        self.assertFalse(df["is_simulated"].iloc[0])
        self.assertAlmostEqual(df["perplexity"].iloc[0], 5.0, places=4)


class CliTests(unittest.TestCase):
    def test_quantize_cli_ptq_saves_model_and_metadata(self):
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as output_dir, \
             patch("bitforge.cli.AutoTokenizer.from_pretrained", return_value=DummyTokenizer()), \
             patch("bitforge.cli.AutoModelForCausalLM.from_pretrained", return_value=SaveableLinearModel()):
            result = runner.invoke(
                main,
                [
                    "quantize",
                    "--model-id",
                    "dummy-model",
                    "--method",
                    "ptq",
                    "--bits",
                    "2",
                    "--output-path",
                    output_dir,
                ],
            )

            metadata_path = os.path.join(output_dir, "bitforge_quantization.json")
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(os.path.exists(metadata_path))
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

        self.assertEqual(metadata["method"], "ptq")
        self.assertEqual(metadata["bits"], 2)
        self.assertEqual(metadata["quantized_module_count"], 1)

    def test_evaluate_cli_reports_real_metric_keys_with_mocked_model(self):
        runner = CliRunner()
        eval_blocks = [{"input_ids": torch.tensor([1, 2, 3, 4]), "attention_mask": torch.ones(4, dtype=torch.long)}]

        with patch("bitforge.cli.AutoTokenizer.from_pretrained", return_value=DummyTokenizer()), \
             patch("bitforge.cli.AutoModelForCausalLM.from_pretrained", return_value=UniformLogitModel(vocab_size=5)), \
             patch("bitforge.cli.CalibrationDataLoader.get_wikitext2", return_value=eval_blocks):
            result = runner.invoke(
                main,
                [
                    "evaluate",
                    "--model-id",
                    "dummy-model",
                    "--dataset",
                    "wikitext2",
                    "--device",
                    "cpu",
                    "--n-samples",
                    "1",
                    "--seq-len",
                    "4",
                    "--max-new-tokens",
                    "2",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('"perplexity"', result.output)
        self.assertIn('"latency"', result.output)
        self.assertIn('"memory"', result.output)


if __name__ == "__main__":
    unittest.main()
