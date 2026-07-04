"""
Merges the trained LoRA adapter into base Phi-3 Mini weights.

Uses bfloat16 rather than FP32 -- confirmed necessary given Colab's
CPU runtime provides only ~11GB available RAM (verified via `free -h`),
while FP32 for a 3.8B-parameter model requires ~15.2GB, which would
reliably exceed available memory and trigger a silent OOM kill (as
observed in the previous attempt). bfloat16 halves memory to ~7.6GB,
comfortably fitting available RAM, and has better native CPU operator
support in PyTorch than float16, making it the correct choice for CPU
inference/merging specifically (not just a memory-driven compromise).
"""

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "model/checkpoints/healthcare_slm_lora/final_adapter"
MERGED_OUTPUT_PATH = "model/checkpoints/healthcare_slm_merged"


def merge_and_save():
    print("Loading base model in bfloat16 on CPU, low_cpu_mem_usage=True...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print("Base model loaded.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("Adapter loaded.")

    print("Merging adapter into base weights...")
    merged_model = peft_model.merge_and_unload()
    print("Merge computation complete.")

    del base_model, peft_model
    gc.collect()

    print(f"Saving merged model to {MERGED_OUTPUT_PATH}...")
    merged_model.save_pretrained(MERGED_OUTPUT_PATH, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_OUTPUT_PATH)

    print("Merge complete.")


if __name__ == "__main__":
    merge_and_save()
