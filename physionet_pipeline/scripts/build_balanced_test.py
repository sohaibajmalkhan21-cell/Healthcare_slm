import json
import random

IN_PATH = "../outputs/test_natural.jsonl"
OUT_PATH = "../outputs/test_balanced.jsonl"

def main():
    examples = [json.loads(l) for l in open(IN_PATH)]
    positives = [e for e in examples if e["label"] == 1]
    negatives = [e for e in examples if e["label"] == 0]
    print(f"Available: {len(positives)} positive, {len(negatives)} negative")

    random.seed(42)
    random.shuffle(negatives)
    negatives = negatives[:len(positives)]  # 1:1 balance
    balanced = positives + negatives
    random.shuffle(balanced)

    with open(OUT_PATH, "w") as f:
        for e in balanced:
            f.write(json.dumps(e) + "\n")
    print(f"Balanced test set: {len(balanced)} examples ({len(positives)} pos, {len(negatives)} neg)")

if __name__ == "__main__":
    main()
