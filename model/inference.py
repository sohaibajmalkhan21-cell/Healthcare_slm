"""
Grounded inference: loads Phi-3 Mini (INT4) and generates answers
constrained to retrieved RAG context, with an explicit no-hallucination
contract in the system prompt.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

SYSTEM_PROMPT = """You are a clinical reference assistant supporting an IoT-based patient monitoring system. You must answer using ONLY the reference material provided below. Follow these rules strictly:

1. Base your answer only on facts and thresholds stated in the provided reference documents.
2. You ARE permitted, and expected, to compare a specific numeric value given in the question against a range or threshold stated in the reference material (for example: if a document states "above 100 bpm indicates tachycardia" and the question gives a heart rate of 145 bpm, you should state that this exceeds the threshold and explain what the document says that implies). This is retrieval-grounded reasoning, not guessing, and you must perform it when the reference material supports it.
3. Only refuse to answer if the reference material genuinely does not contain a relevant threshold, range, or fact needed to address the question. Do not refuse merely because the exact conclusion is not spelled out verbatim if it follows directly from a stated threshold.
4. When you state a fact or conclusion, mention which document title it is grounded in.
5. Do not introduce any medical fact, threshold, or number that is not present in the reference material.
6. Be concise and clinically precise. This output may inform patient care decisions, so flag concerning combinations of values clearly."""


class GroundedInferenceEngine:
    """Loads Phi-3 Mini in 4-bit precision and generates RAG-grounded answers."""

    def __init__(self, model_name: str = MODEL_NAME):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
        )

    def _build_prompt(self, query: str, retrieved_docs: list[dict]) -> list[dict]:
        context_block = "\n\n".join(
            f"[{doc['title']}]\n{doc['content']}" for doc in retrieved_docs
        )
        user_message = f"Reference material:\n\n{context_block}\n\nQuestion: {query}"

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

    def generate(self, query: str, retrieved_docs: list[dict], max_new_tokens: int = 256) -> str:
        messages = self._build_prompt(query, retrieved_docs)
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
    from rag.retriever import MedicalRetriever

    print("Loading retriever...")
    retriever = MedicalRetriever("data/knowledge_base/vitals_reference.json")

    print("Loading Phi-3 Mini (INT4)... this may take a minute on first run.")
    engine = GroundedInferenceEngine()

    test_query = "A patient's heart rate is 145 bpm and SpO2 is 88%. Is this concerning?"
    print(f"\nQuery: {test_query}")

    docs = retriever.retrieve(test_query, k=3)
    print(f"\nRetrieved {len(docs)} documents:")
    for d in docs:
        print(f"  - {d['title']} (score: {d['similarity_score']:.3f})")

    answer = engine.generate(test_query, docs)
    print(f"\nGenerated answer:\n{answer}")
