# Training Results — Hopper-Medium-v2

Trained on Google Colab T4 GPU.
Date: 2026-05-01
Steps: 100,000 | Batch: 1024 | Device: cuda

## Final Normalized Scores

| Agent | Our Score | Paper Score |
|---|---|---|
| TD3+BC | 27.9 | 59.3 |
| CQL | 106.4 | 79.4 |
| IQL | 80.9 | 75.1 |

## Plots
- `plots/dataset_exploration.png` — reward/action/return distributions
- `plots/learning_curves.png`    — normalized score, Q-values, CQL penalty
- `plots/final_scores.png`       — bar chart vs paper baselines
- `plots/expectile_theory.png`   — IQL expectile regression visualization
- `plots/cql_ood_analysis.png`   — CQL conservative Q-value analysis
