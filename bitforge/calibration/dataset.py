import os
from typing import Iterable, List, Optional

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer


class CalibrationDataLoader:
    """
    Loads and processes datasets for quantizer calibration.
    Supports standard datasets (Wikitext2, C4, PTB) and custom domain specific datasets.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def get_wikitext2(self, n_samples: int = 128, split: str = "train") -> List[dict]:
        """
        Load Wikitext-2 and slice it into uniform blocks of size seq_len.
        """
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        return self._tokenize_dataset(dataset, n_samples=n_samples, text_column="text")

    def get_c4(self, n_samples: int = 128, split: str = "train") -> List[dict]:
        """
        Load C4 clean dataset for generic calibration.
        """
        dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)
        return self._tokenize_dataset(dataset, n_samples=n_samples, text_column="text")

    def get_custom_dataset(
        self,
        path_or_name: str,
        n_samples: int = 128,
        split: str = "train",
        text_column: Optional[str] = None,
    ) -> List[dict]:
        """
        Load a custom user dataset (e.g. medical, legal, code) to observe out-of-domain
        calibration impacts on general reasoning vs specialized task performance.
        """
        if os.path.isfile(path_or_name):
            dataset = load_dataset("text", data_files=path_or_name, split="train")
        else:
            dataset = load_dataset(path_or_name, split=split)

        return self._tokenize_dataset(dataset, n_samples=n_samples, text_column=text_column)

    def get_from_texts(self, texts: Iterable[str], n_samples: int = 128) -> List[dict]:
        """
        Convert raw text strings into fixed-length token blocks.

        This is useful for tests, notebooks, and callers that already have text in memory.
        """
        return self._tokenize_texts(texts, n_samples=n_samples)

    def _tokenize_dataset(
        self,
        dataset,
        n_samples: int,
        text_column: Optional[str] = None,
    ) -> List[dict]:
        def iter_texts():
            for row in dataset:
                column = text_column or self._infer_text_column(row)
                text = row.get(column, "")
                if isinstance(text, str):
                    yield text

        return self._tokenize_texts(iter_texts(), n_samples=n_samples)

    def _tokenize_texts(self, texts: Iterable[str], n_samples: int) -> List[dict]:
        blocks: List[dict] = []
        token_buffer: List[int] = []

        for text in texts:
            if len(blocks) >= n_samples:
                break

            text = text.strip()
            if not text:
                continue

            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            input_ids = encoded["input_ids"]
            if not input_ids:
                continue

            token_buffer.extend(input_ids)
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                token_buffer.append(eos_token_id)

            while len(token_buffer) >= self.seq_len and len(blocks) < n_samples:
                block_ids = token_buffer[: self.seq_len]
                token_buffer = token_buffer[self.seq_len :]
                blocks.append(self._make_block(block_ids))

        if token_buffer and len(blocks) < n_samples:
            blocks.append(self._make_block(self._pad_to_seq_len(token_buffer)))

        return blocks

    def _make_block(self, input_ids: List[int]) -> dict:
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        return {
            "input_ids": input_tensor,
            "attention_mask": torch.ones_like(input_tensor),
        }

    def _pad_to_seq_len(self, input_ids: List[int]) -> List[int]:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0

        return input_ids + [pad_token_id] * (self.seq_len - len(input_ids))

    @staticmethod
    def _infer_text_column(row: dict) -> str:
        for column, value in row.items():
            if isinstance(value, str):
                return column
        raise ValueError("Dataset row does not contain a string text column.")
