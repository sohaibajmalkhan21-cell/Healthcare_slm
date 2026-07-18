import json
import random
import pandas as pd
from sare import build_summary_for_patient_hour

IN_PATH = "../outputs/patients_combined.parquet"
TRAIN_OUT = "../outputs/train.jsonl"
VAL_OUT = "../outputs/val.jsonl"
TEST_OUT = "../outputs/test.jsonl"

SYSTEM_PROMPT = "You are a clinical reference assistant. Based on the described vital sign trends, predict whether the patient is likely to develop sepsis within the next 6 hours. Answer with 'yes' or 'no' only."

def build_examples(df: pd.DataFrame, min_hour: int = 6):
    random.seed(42)
    positives, negatives = [], []
    for pid, pdf in df.groupby("patient_id"):
        pdf = pdf.sort_values("hour").reset_index(drop=True)
        for hour in range(min_hour, len(pdf)):
            label = int(pdf.loc[hour, "SepsisLabel"])
            summary = build_summary_for_patient_hour(pdf, hour)
            if summary is None:
                continue
            ex = {
                "system": SYSTEM_PROMPT, "context": summary,
                "question": "Is sepsis likely within the next 6 hours?",
                "response": "yes" if label == 1 else "no", "label": label,
                "patient_id": pid, "hour": int(hour),
            }
            (positives if label == 1 else negatives).append(ex)

    print(f"Raw positives: {len(positives)}, raw negatives: {len(negatives)}")
    # Balance: keep all positives, downsample negatives to 3x positives
    # (near-1:1 hurts specificity learning; pure natural ratio ~1:70 causes
    # model to always predict "no" -- 3:1 is a standard compromise)
    random.shuffle(negatives)
    negatives = negatives[:len(positives) * 3]
    examples = positives + negatives
    random.shuffle(examples)
    print(f"Balanced set: {len(positives)} positive, {len(negatives)} negative")
    return examples

def stratified_split(examples, train_frac=0.7, val_frac=0.15):
    patient_ids = list({e["patient_id"] for e in examples})
    random.seed(42)
    random.shuffle(patient_ids)
    n = len(patient_ids)
    train_ids = set(patient_ids[:int(n*train_frac)])
    val_ids = set(patient_ids[int(n*train_frac):int(n*(train_frac+val_frac))])
    train, val, test = [], [], []
    for e in examples:
        if e["patient_id"] in train_ids:
            train.append(e)
        elif e["patient_id"] in val_ids:
            val.append(e)
        else:
            test.append(e)
    return train, val, test

def save_jsonl(examples, path):
    with open(path, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")

if __name__ == "__main__":
    df = pd.read_parquet(IN_PATH)
    examples = build_examples(df)
    train, val, test = stratified_split(examples)
    save_jsonl(train, TRAIN_OUT)
    save_jsonl(val, VAL_OUT)
    save_jsonl(test, TEST_OUT)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
