"""
Synthetic + real instruction-tuning dataset generator for the
healthcare IoT SLM fine-tuning pass (Milestone 5).

Combines four synthetic categories generated from the project's own
knowledge base (targeting specific behaviors identified during
Milestone 4 evaluation) with a supplementary sample of real PubMedQA
examples (Jin et al., 2019, EMNLP) for broader biomedical reasoning
exposure. Produces a stratified train/validation split.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, ".")

SYSTEM_PROMPT = """You are a clinical reference assistant supporting an IoT-based patient monitoring system. You must answer using ONLY the reference material provided below. Follow these rules strictly:

1. Base your answer only on facts and thresholds stated in the provided reference documents.
2. You ARE permitted, and expected, to compare a specific numeric value given in the question against a range or threshold stated in the reference material. This is retrieval-grounded reasoning, not guessing, and you must perform it when the reference material supports it.
3. Only refuse to answer if the reference material genuinely does not contain a relevant threshold, range, or fact needed to address the question.
4. When you state a fact or conclusion, mention which document title it is grounded in.
5. Do not introduce any medical fact, threshold, or number that is not present in the reference material.
6. Be concise and clinically precise."""

QUESTION_TEMPLATES = {
    "vitals-hr-001": [
        "What is a normal heart rate for an adult?",
        "What heart rate range is considered healthy at rest?",
        "At what point is a heart rate considered too high or too low?",
    ],
    "vitals-spo2-001": [
        "What is a normal blood oxygen saturation level?",
        "What SpO2 reading is considered healthy?",
        "At what oxygen saturation level should a patient be concerned?",
    ],
    "vitals-temp-001": [
        "What is a normal body temperature?",
        "What temperature is considered a fever?",
        "At what temperature is hypothermia diagnosed?",
    ],
    "vitals-rr-001": [
        "What is a normal respiratory rate for an adult?",
        "How many breaths per minute is considered healthy?",
        "What breathing rate indicates respiratory distress?",
    ],
    "vitals-bp-001": [
        "What is considered normal blood pressure?",
        "At what blood pressure reading is someone diagnosed with hypertension?",
        "What blood pressure level indicates hypotension?",
    ],
    "vitals-alert-001": [
        "Why is it important to monitor multiple vital signs together?",
        "Is one abnormal vital sign as concerning as multiple abnormal readings?",
        "How should multiple abnormal vitals be prioritized clinically?",
    ],
}


def load_knowledge_base(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_grounded_qa_examples(documents: list[dict]) -> list[dict]:
    """Category 1: Direct grounded QA -- paraphrased questions per document."""
    examples = []
    for doc in documents:
        for question in QUESTION_TEMPLATES.get(doc["id"], []):
            context_block = f"[{doc['title']}]\n{doc['content']}"
            response = (
                f"According to the reference document \"{doc['title']}\", "
                f"{doc['content'][0].lower()}{doc['content'][1:]}"
            )
            examples.append({
                "system": SYSTEM_PROMPT, "context": context_block,
                "question": question, "response": response,
                "category": "grounded_qa", "source_doc_id": doc["id"],
            })
    return examples


def generate_threshold_comparison_examples(documents: list[dict], n_per_doc: int = 20, seed: int = 42) -> list[dict]:
    """Category 2: Threshold-comparison reasoning -- randomized synthetic values."""
    rng = random.Random(seed)
    vital_specs = [
        ("vitals-hr-001", "heart rate", "bpm", 60, 100, 35, 180),
        ("vitals-spo2-001", "SpO2", "%", 95, 100, 75, 100),
        ("vitals-temp-001", "temperature", "C", 36.1, 37.2, 33.0, 41.0),
        ("vitals-rr-001", "respiratory rate", "breaths/min", 12, 20, 5, 40),
    ]
    doc_lookup = {doc["id"]: doc for doc in documents}
    examples = []

    for doc_id, vital_name, unit, low, high, plaus_low, plaus_high in vital_specs:
        doc = doc_lookup[doc_id]
        context_block = f"[{doc['title']}]\n{doc['content']}"

        for _ in range(n_per_doc):
            bucket = rng.choice(["normal", "high", "low", "borderline"])
            if bucket == "normal":
                value = round(rng.uniform(low, high), 1)
            elif bucket == "high":
                value = round(rng.uniform(high + (high - low) * 0.1, plaus_high), 1)
            elif bucket == "low":
                value = round(rng.uniform(plaus_low, low - (high - low) * 0.1), 1)
            else:
                offset = (high - low) * 0.05
                value = round(rng.choice([low - offset, high + offset]), 1)

            question = f"A patient's {vital_name} is {value} {unit}. Is this within the normal range?"

            if low <= value <= high:
                response = (
                    f"According to \"{doc['title']}\", a normal {vital_name} range is "
                    f"{low} to {high} {unit}. A reading of {value} {unit} falls within this normal range."
                )
            elif value > high:
                response = (
                    f"According to \"{doc['title']}\", a normal {vital_name} range is "
                    f"{low} to {high} {unit}. A reading of {value} {unit} exceeds this range, "
                    f"which the reference material indicates may warrant clinical evaluation."
                )
            else:
                response = (
                    f"According to \"{doc['title']}\", a normal {vital_name} range is "
                    f"{low} to {high} {unit}. A reading of {value} {unit} falls below this range, "
                    f"which the reference material indicates may warrant clinical evaluation."
                )

            examples.append({
                "system": SYSTEM_PROMPT, "context": context_block,
                "question": question, "response": response,
                "category": "threshold_comparison", "source_doc_id": doc_id,
            })
    return examples


def generate_calibrated_refusal_examples(documents: list[dict], seed: int = 42) -> list[dict]:
    """Category 3: Calibrated refusal -- genuinely out-of-scope questions."""
    rng = random.Random(seed)
    out_of_scope_questions = [
        "What medication should be given for a heart rate of 145 bpm?",
        "What is the recommended dosage of oxygen therapy for low SpO2?",
        "Should this patient be prescribed antipyretics for their fever?",
        "What is the diagnosis for a patient with these vital signs?",
        "How long will it take for this patient's condition to improve?",
        "What is the patient's prognosis given these readings?",
        "Should this patient be admitted to the ICU?",
        "What antibiotic would be appropriate for this patient's fever?",
        "Is this patient's abnormal heart rate caused by anxiety or a cardiac condition?",
        "What follow-up tests should be ordered based on these vitals?",
    ]
    refusal_response = (
        "The available reference material does not contain sufficient information "
        "to answer this. The retrieved documents describe normal vital sign ranges "
        "and general escalation guidance, but do not include medication, diagnostic, "
        "treatment, or prognostic information. This question should be directed to "
        "a qualified clinician."
    )
    examples = []
    for question in out_of_scope_questions:
        doc = rng.choice(documents)
        context_block = f"[{doc['title']}]\n{doc['content']}"
        examples.append({
            "system": SYSTEM_PROMPT, "context": context_block,
            "question": question, "response": refusal_response,
            "category": "calibrated_refusal", "source_doc_id": doc["id"],
        })
    return examples


def generate_vitals_triggered_examples(documents: list[dict], n_examples: int = 60, seed: int = 42) -> list[dict]:
    """Category 4: Vitals-triggered queries -- uses the real VitalsSimulator."""
    from edge.vitals_simulator import VitalsSimulator

    doc_lookup = {doc["id"]: doc for doc in documents}
    hr_doc, spo2_doc = doc_lookup["vitals-hr-001"], doc_lookup["vitals-spo2-001"]
    temp_doc, rr_doc = doc_lookup["vitals-temp-001"], doc_lookup["vitals-rr-001"]
    alert_doc = doc_lookup["vitals-alert-001"]

    sim = VitalsSimulator(anomaly_probability=0.4, seed=seed)
    examples = []

    for _ in range(n_examples):
        reading = sim.next_reading()
        question = (
            f"IoT sensor reading received: heart rate {reading.heart_rate_bpm} bpm, "
            f"SpO2 {reading.spo2_percent}%, temperature {reading.temperature_celsius}C, "
            f"respiratory rate {reading.respiratory_rate_bpm} breaths/min. Evaluate this reading."
        )

        abnormal_flags = []
        if not (60 <= reading.heart_rate_bpm <= 100):
            abnormal_flags.append(("heart rate", reading.heart_rate_bpm, "bpm", hr_doc))
        if not (95 <= reading.spo2_percent <= 100):
            abnormal_flags.append(("SpO2", reading.spo2_percent, "%", spo2_doc))
        if not (36.1 <= reading.temperature_celsius <= 37.2):
            abnormal_flags.append(("temperature", reading.temperature_celsius, "C", temp_doc))
        if not (12 <= reading.respiratory_rate_bpm <= 20):
            abnormal_flags.append(("respiratory rate", reading.respiratory_rate_bpm, "breaths/min", rr_doc))

        relevant_docs = [hr_doc, spo2_doc, temp_doc, rr_doc]
        if len(abnormal_flags) >= 2:
            relevant_docs.append(alert_doc)
        context_block = "\n\n".join(f"[{d['title']}]\n{d['content']}" for d in relevant_docs)

        if not abnormal_flags:
            response = (
                "All four vital signs fall within normal ranges according to the reference "
                "material: heart rate, SpO2, temperature, and respiratory rate are each within "
                "their respective normal thresholds. No escalation is indicated."
            )
        else:
            parts = [
                f"{name} of {value} {unit} falls outside the normal range stated in \"{doc['title']}\""
                for name, value, unit, doc in abnormal_flags
            ]
            if len(abnormal_flags) >= 2:
                response = (
                    f"Multiple abnormal readings detected: {'; '.join(parts)}. "
                    f"According to \"{alert_doc['title']}\", a combination of abnormal vitals "
                    f"is more clinically significant than any single abnormal reading in isolation, "
                    f"and this combination warrants prompt escalation."
                )
            else:
                response = (
                    f"One abnormal reading detected: {parts[0]}. This single deviation warrants "
                    f"monitoring per the reference material, though the other vitals remain within normal ranges."
                )

        examples.append({
            "system": SYSTEM_PROMPT, "context": context_block,
            "question": question, "response": response,
            "category": "vitals_triggered", "source_doc_id": "multiple",
        })
    return examples


def load_pubmedqa_examples(n_examples: int = 25, seed: int = 42) -> list[dict]:
    """Category 5: Real PubMedQA examples (Jin et al., 2019, EMNLP) for broader biomedical exposure."""
    from datasets import load_dataset

    print("Loading PubMedQA (pqa_labeled) from Hugging Face...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    selected_indices = indices[:n_examples]

    examples = []
    for idx in selected_indices:
        row = dataset[idx]
        contexts, labels = row["context"]["contexts"], row["context"]["labels"]
        context_block = "\n\n".join(f"[{label}]\n{text}" for label, text in zip(labels, contexts))
        response = f"{row['long_answer']} (Summary judgment: {row['final_decision']})."

        examples.append({
            "system": SYSTEM_PROMPT, "context": context_block,
            "question": row["question"], "response": response,
            "category": "pubmedqa_general", "source_doc_id": f"pubmedqa_{idx}",
        })

    print(f"Loaded {len(examples)} PubMedQA examples")
    return examples


def train_val_split(examples: list[dict], val_fraction: float = 0.15, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Stratified split -- every category represented in both train and validation."""
    rng = random.Random(seed)
    by_category: dict[str, list[dict]] = {}
    for ex in examples:
        by_category.setdefault(ex["category"], []).append(ex)

    train, val = [], []
    for category, cat_examples in by_category.items():
        shuffled = cat_examples[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_fraction))
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def run_final_generation():
    """Generate all five categories (4 synthetic + PubMedQA), split, and save."""
    documents = load_knowledge_base("data/knowledge_base/vitals_reference.json")

    grounded_qa = generate_grounded_qa_examples(documents)
    threshold_comparison = generate_threshold_comparison_examples(documents, n_per_doc=20)
    calibrated_refusal = generate_calibrated_refusal_examples(documents)
    vitals_triggered = generate_vitals_triggered_examples(documents, n_examples=60)
    pubmedqa = load_pubmedqa_examples(n_examples=25)

    all_examples = grounded_qa + threshold_comparison + calibrated_refusal + vitals_triggered + pubmedqa

    print(f"\nCategory 1 (grounded QA):          {len(grounded_qa):>4}")
    print(f"Category 2 (threshold comparison): {len(threshold_comparison):>4}")
    print(f"Category 3 (calibrated refusal):   {len(calibrated_refusal):>4}")
    print(f"Category 4 (vitals-triggered):     {len(vitals_triggered):>4}")
    print(f"Category 5 (PubMedQA):             {len(pubmedqa):>4}")
    print(f"{'-'*45}")
    print(f"Total examples:                    {len(all_examples):>4}")

    train, val = train_val_split(all_examples, val_fraction=0.15)
    print(f"\nTrain set: {len(train)} | Validation set: {len(val)}")

    train_by_cat, val_by_cat = {}, {}
    for ex in train:
        train_by_cat[ex["category"]] = train_by_cat.get(ex["category"], 0) + 1
    for ex in val:
        val_by_cat[ex["category"]] = val_by_cat.get(ex["category"], 0) + 1

    print("\nPer-category split:")
    for cat in train_by_cat:
        print(f"  {cat:<22} train: {train_by_cat.get(cat, 0):>3}  val: {val_by_cat.get(cat, 0):>3}")

    Path("data/training").mkdir(parents=True, exist_ok=True)
    with open("data/training/train.jsonl", "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    with open("data/training/val.jsonl", "w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(ex) + "\n")

    print(f"\nSaved train.jsonl ({len(train)}) and val.jsonl ({len(val)})")
    return train, val


if __name__ == "__main__":
    run_final_generation()
