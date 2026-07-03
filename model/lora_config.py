"""
LoRA/QLoRA configuration for fine-tuning Phi-3 Mini.

Configuration decisions (see Milestone 5 discussion for full justification):
- rank (r) = 16: sufficient adaptation capacity for behavior-shaping
  (threshold comparison, calibrated refusal) without overfitting risk
  on our 193-example dataset. Per Hu et al. (2021), intrinsic rank of
  needed updates is often far lower than full parameter count suggests.
- alpha = 32 (2x rank): standard scaling ratio balancing adapter
  influence against catastrophic forgetting of base model capabilities.
- target_modules = all linear layers (attention q/k/v/o + MLP gate/up/down):
  per QLoRA ablations (Dettmers et al., 2023), broader coverage improves
  reasoning-behavior adaptation quality vs. attention-only LoRA.
- dropout = 0.05: light regularization, appropriate given our modest
  dataset size and the presence of a validation set for monitoring.
"""

from peft import LoraConfig, TaskType


def get_lora_config() -> LoraConfig:
    """
    Returns the LoRA configuration for Phi-3 Mini fine-tuning.

    Target module names match Phi-3's internal naming convention for
    its attention and MLP projection layers.
    """
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "qkv_proj",      # Phi-3 uses a fused QKV projection, not separate q/k/v
            "o_proj",        # attention output projection
            "gate_up_proj",  # Phi-3 uses a fused gate+up projection in its MLP
            "down_proj",     # MLP down projection
        ],
        lora_dropout=0.05,
        bias="none",              # standard practice: don't adapt bias terms, minimal benefit
        task_type=TaskType.CAUSAL_LM,
    )


if __name__ == "__main__":
    config = get_lora_config()
    print("LoRA Configuration:")
    print(f"  Rank (r):           {config.r}")
    print(f"  Alpha:              {config.lora_alpha}")
    print(f"  Scaling factor:     {config.lora_alpha / config.r}")
    print(f"  Target modules:     {config.target_modules}")
    print(f"  Dropout:            {config.lora_dropout}")
    print(f"  Task type:          {config.task_type}")
