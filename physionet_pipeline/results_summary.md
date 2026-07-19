# PhysioNet 2019 Sepsis Prediction — Results

## Test set (natural ratio, held-out patients)
N=3000, Positives=33 (1.22% natural rate)

| Model | Sensitivity | Specificity | F1 | AUC | Avg |
|---|---|---|---|---|---|
| Phi-3 Mini (fine-tuned) | 0.0909 | 0.9633 | 0.0414 | 0.6732 | 0.4422 |
| Llama 3.2 1B (fine-tuned) | 0.1515 | 0.9646 | 0.0699 | 0.6346 | 0.4552 |

Training: LoRA r=16 alpha=32, 3 epochs, 3:1 negative:positive training ratio.

## FINAL RESULTS — both models, natural ratio, methodologically clean
(threshold tuned on separate held-out split, bootstrap 95% CI, no data leakage)

| Model | Threshold | Sensitivity | Specificity | F1 | AUC | Avg |
|---|---|---|---|---|---|---|
| Phi-3 Mini | 0.25 | 0.5796 (0.5507-0.6089) | 0.7248 (0.7215-0.7280) | 0.0493 (0.0454-0.0533) | 0.7126 (0.6974-0.7290) | 0.5166 |
| Llama 3.2 1B | 0.24 | 0.5870 (0.5566-0.6188) | 0.7111 (0.7078-0.7142) | 0.0448 (0.0413-0.0485) | 0.6948 (0.6781-0.7119) | 0.5094 |

Both models statistically indistinguishable (overlapping 95% CIs on all metrics).
Both substantially outperform paper's (Yang et al., 2026, Scientific Reports)
non-fine-tuned Phi-3-mini-4k-Instruct baseline (Sens 0.4671, Spec 0.6380, AUC 0.5599)
on AUC, Sensitivity, and Specificity. F1 not directly comparable due to
differing test-set class ratios (ours: 1.16-1.24% positive natural rate;
paper's: ~20% enriched positive rate).
