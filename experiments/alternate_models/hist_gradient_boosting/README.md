# Hist Gradient Boosting — Experiment Record

## Algorithm Family
Boosting

## Description
Scikit-learn's histogram-based gradient boosting (similar to LightGBM). Builds trees sequentially, each correcting the residuals of the previous. Handles missing values natively.

## Configuration
```
{
  "max_iter": 300,
  "learning_rate": 0.1,
  "max_leaf_nodes": 31,
  "random_state": 42
}
```

## How to Reproduce
```bash
python scripts/benchmark_classifiers.py hist_gradient_boosting --force
```

## Recorded Results

### Cross-Validation (5-fold GroupKFold, 80 training trips)
| Metric | Value |
|---|---|
| Macro F1 (mean) | 0.9184 |
| Top-1 Accuracy (mean) | 0.8732 |
| Log Loss (mean) | 0.0887 |

### Student-Testing Partition (10 trips)
| Metric | Value |
|---|---|
| Top-1 Accuracy | 0.8873 |
| Macro F1 | 0.9324 |
| Precision (macro) | 0.9361 |
| Recall (macro) | 0.9289 |
| Log Loss | 0.0700 |
| Route Continuity | 0.9526 |
| Candidate Miss Rate | 0.0010 |

### Timing
| | Value |
|---|---|
| Fit time (full train) | 3.8s |
| Inference | 0.41 ms / 1k rows |

## Notes
This model was not selected as the final model. See the comparison table in `results/models/comparison.csv` for the full ranking.
