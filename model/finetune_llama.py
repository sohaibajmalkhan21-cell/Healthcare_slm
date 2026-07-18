import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)
from peft import get_peft_model, prepare_model_for_kbit_training
from model.lora_config_llama import get_lora_config
from model.dataset_utils import GroundedSFTDataset

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
OUTPUT_DIR = "model/checkpoints/llama1b_lora"

def load_quantized_base_model():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=quant_config, device_map="auto",
    )
    return model, tokenizer

def build_peft_model(base_model):
    base_model = prepare_model_for_kbit_training(base_model)
    lora_config = get_lora_config()
    peft_model = get_peft_model(base_model, lora_config)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"Trainable params: {trainable:,} ({100*trainable/total:.3f}%)")
    return peft_model

def train():
    print("Loading quantized base model...")
    base_model, tokenizer = load_quantized_base_model()
    print("Attaching LoRA adapter...")
    peft_model = build_peft_model(base_model)

    print("Loading datasets...")
    train_dataset = GroundedSFTDataset("data/training/train.jsonl", tokenizer)
    val_dataset = GroundedSFTDataset("data/training/val.jsonl", tokenizer)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5,
        fp16=True,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting training...\n")
    trainer.train()

    print("\nSaving final adapter...")
    peft_model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    print(f"Saved to {OUTPUT_DIR}/final_adapter")

if __name__ == "__main__":
    train()
