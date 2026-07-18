"""
LoRA configuration for fine-tuning Llama 3.2 1B.
Same r=16, alpha=32, dropout=0.05 as Phi-3 config for fair comparison.
target_modules differ: Llama uses separate q/k/v/o and gate/up/down
projections, not Phi-3's fused qkv_proj/gate_up_proj.
"""
from peft import LoraConfig, TaskType

def get_lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

if __name__ == "__main__":
    config = get_lora_config()
    print("Rank:", config.r, "Alpha:", config.lora_alpha, "Targets:", config.target_modules)
