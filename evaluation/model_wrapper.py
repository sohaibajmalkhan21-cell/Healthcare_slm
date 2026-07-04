"""
Unified inference wrapper for evaluation: loads Phi-3 Mini in 4-bit,
optionally attaching the fine-tuned LoRA adapter, so base and
fine-tuned behavior can be compared under identical conditions.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "model/checkpoints/healthcare_slm_lora/final_adapter"

SYSTEM_PROMPT = """You are a clinical reference assistant supporting an IoT-based patient monitoring system. You must answer using ONLY the reference material provided below. Follow these rules strictly:

1. Base your answer only on facts and thresholds stated in the provided reference documents.
2. You ARE permitted, and expected, to compare a specific numeric value given in the question against a range or threshold stated in the reference material. This is retrieval-grounded reasoning, not guessing, and you must perform it when the reference material supports it.
3. Only refuse to answer if the reference material genuinely does not contain a relevant threshold, range, or fact needed to address the question.
4. When you state a fact or conclusion, mention which document title it is grounded in.
5. Do not introduce any medical fact, threshold, or number that is not present in the reference material.
6. Be concise and clinically precise."""


class EvaluationModel:
    """
    Loads Phi-3 Mini in 4-bit, with an option to attach the LoRA
    adapter. use_adapter=False gives the pure baseline behavior
    (identical to Milestone 4's inference engine); use_adapter=True
    gives the fine-tuned behavior -- letting us A/B compare directly.
    """

    def __init__(self, use_adapter: bool = True):
        self.use_adapter = use_adapter

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # correct for generation, per Milestone 5's lesson

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quant_config,
            device_map="auto",
        )

        if use_adapter:
            print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
            self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
            print("Adapter attached.")
        else:
            print("Using base model only (no adapter) -- baseline configuration.")
            self.model = base_model

        self.model.eval()

    def generate(self, question: str, context: str, max_new_tokens: int = 200) -> str:
        user_content = f"Reference material:\n\n{context}\n\nQuestion: {question}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return generated_text.strip()


if __name__ == "__main__":
    # Quick sanity test: the exact Milestone 4 over-refusal case,
    # run through the fine-tuned model, to see the first real signal
    # of whether fine-tuning improved on the original failure.
    test_context = (
        "[Normal Resting Heart Rate Range (Adult)]\n"
        "For a healthy adult at rest, a normal heart rate ranges from 60 to 100 beats per minute (bpm). "
        "Heart rates consistently below 60 bpm (bradycardia) or above 100 bpm (tachycardia) at rest may "
        "indicate an underlying condition and typically warrant clinical evaluation.\n\n"
        "[Normal Blood Oxygen Saturation (SpO2)]\n"
        "Normal SpO2 (peripheral oxygen saturation) for a healthy individual is typically 95% to 100%. "
        "Readings at or below 90% indicate significant hypoxemia and require prompt clinical attention."
    )
    test_question = "A patient's heart rate is 145 bpm and SpO2 is 88%. Is this concerning?"

    print("=" * 60)
    print("Loading FINE-TUNED model (with adapter)...")
    print("=" * 60)
    ft_model = EvaluationModel(use_adapter=True)

    print(f"\nQuestion: {test_question}\n")
    ft_answer = ft_model.generate(test_question, test_context)
    print(f"Fine-tuned answer:\n{ft_answer}")
