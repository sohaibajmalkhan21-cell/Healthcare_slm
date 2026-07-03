"""
Dataset loading and tokenization for LoRA fine-tuning.

Formats each training example into Phi-3's chat template, then masks
prompt tokens (system + context + question) from the loss calculation
so the model only learns to predict the response.

IMPORTANT: padding_side must be "right" for training. Most causal LM
tokenizers (including Phi-3's) default to "left" padding, which is
correct for generation but breaks label masking here, since our
masking logic assumes prompt tokens occupy the start of the sequence
and padding occupies the end. Left-padding silently masks padding
tokens instead of the prompt, leaving the model trained on the wrong
objective with no visible error -- caught via the decode sanity check.
"""

import json
import torch
from torch.utils.data import Dataset


class GroundedSFTDataset(Dataset):
    """
    Wraps our train.jsonl / val.jsonl files for supervised fine-tuning.

    Each example is formatted as a Phi-3 chat sequence:
      [system] -> [user: context + question] -> [assistant: response]
    with loss masked out everywhere except the assistant's response.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer

        # Force right-padding for training -- see module docstring.
        # This must happen here, not just assumed set by the caller,
        # so this dataset class is safe regardless of how the tokenizer
        # was configured elsewhere (e.g. for inference in Milestone 4,
        # which correctly uses left-padding for generation).
        self.tokenizer.padding_side = "right"

        self.max_length = max_length
        self.examples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]

        user_content = f"Reference material:\n\n{example['context']}\n\nQuestion: {example['question']}"

        messages = [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": user_content},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + example["response"] + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(
            prompt_text, add_special_tokens=False, truncation=True, max_length=self.max_length
        )["input_ids"]

        full_encoding = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        input_ids = full_encoding["input_ids"]
        attention_mask = full_encoding["attention_mask"]

        labels = list(input_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100
        for i, mask_val in enumerate(attention_mask):
            if mask_val == 0:
                labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer padding_side BEFORE dataset init: {tokenizer.padding_side}")

    dataset = GroundedSFTDataset("data/training/train.jsonl", tokenizer)

    print(f"Tokenizer padding_side AFTER dataset init: {tokenizer.padding_side}")
    print(f"Dataset size: {len(dataset)} examples")

    sample = dataset[0]
    print(f"\nSample shapes: input_ids {sample['input_ids'].shape}, "
          f"attention_mask {sample['attention_mask'].shape}, labels {sample['labels'].shape}")

    n_masked = (sample["labels"] == -100).sum().item()
    n_total = len(sample["labels"])
    print(f"Masked tokens (prompt + padding): {n_masked} / {n_total}")
    print(f"Response tokens (trainable): {n_total - n_masked}")

    decoded_response_only = tokenizer.decode(
        [t for t, l in zip(sample["input_ids"].tolist(), sample["labels"].tolist()) if l != -100],
        skip_special_tokens=True
    )
    print(f"\nDecoded response-only tokens (sanity check):\n{decoded_response_only}")

    assert "clinical reference assistant" not in decoded_response_only, (
        "MASKING BUG: system prompt leaked into trainable region!"
    )
    assert "Reference material:" not in decoded_response_only, (
        "MASKING BUG: context leaked into trainable region!"
    )
    print("\n✓ Masking verified correct: no prompt/context text in trainable region.")
