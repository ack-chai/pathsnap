# Lightgbm — Experiment Record

## Algorithm Family
Boosting

## Description
Gradient boosted trees with leaf-wise growth and histogram binning. Very fast and memory-efficient on large datasets. Supports subsample and column subsampling.

## Configuration
```
{
  "n_estimators": 400,
  "learning_rate": 0.05,
  "num_leaves": 63,
  "subsample": 0.8,
  "subsample_freq": 1,
  "colsample_bytree": 0.8,
  "random_state": 42,
  "n_jobs": -1,
  "verbose": -1
}
```

## How to Reproduce
```bash
python scripts/benchmark_classifiers.py lightgbm --force
```

## Recorded Results

### Cross-Validation (5-fold GroupKFold, 80 training trips)
| Metric | Value |
|---|---|
| Macro F1 (mean) | 0.9256 |
| Top-1 Accuracy (mean) | 0.8743 |
| Log Loss (mean) | 0.0821 |

### Student-Testing Partition (10 trips)
| Metric | Value |
|---|---|
| Top-1 Accuracy | 0.8900 |
| Macro F1 | 0.9349 |
| Precision (macro) | 0.9372 |
| Recall (macro) | 0.9327 |
| Log Loss | 0.0755 |
| Route Continuity | 0.9436 |
| Candidate Miss Rate | 0.0010 |

### Timing
| | Value |
|---|---|
| Fit time (full train) | 10.2s |
| Inference | 2.54 ms / 1k rows |

## Notes
This model was not selected as the final model. See the comparison table in `results/models/comparison.csv` for the full ranking.
