# PhysioNet 2019 Sepsis Prediction — Results

## Test set (natural ratio, held-out patients)
N=3000, Positives=33 (1.22% natural rate)

| Model | Sensitivity | Specificity | F1 | AUC | Avg |
|---|---|---|---|---|---|
| Phi-3 Mini (fine-tuned) | 0.0909 | 0.9633 | 0.0414 | 0.6732 | 0.4422 |
| Llama 3.2 1B (fine-tuned) | 0.1515 | 0.9646 | 0.0699 | 0.6346 | 0.4552 |

Training: LoRA r=16 alpha=32, 3 epochs, 3:1 negative:positive training ratio.
