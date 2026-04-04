# Training Results — Hopper-Medium-v2

Trained on Google Colab T4 GPU.  
Date: 2026-04-04  
Steps: 50,000 | Batch: 1024 | Device: cuda

## Final Normalized Scores

| Agent | Our Score | Paper Score |
|---|---|---|
| TD3+BC | 29.2 | 59.3 |
| CQL | 95.9 | 79.4 |
| IQL | -0.4 | 75.1 |

## Plots
- `plots/dataset_exploration.png` — reward/action/return distributions
- `plots/learning_curves.png`    — normalized score, Q-values, CQL penalty
- `plots/final_scores.png`       — bar chart vs paper baselines
- `plots/expectile_theory.png`   — IQL expectile regression visualization
- `plots/cql_ood_analysis.png`   — CQL conservative Q-value analysis
