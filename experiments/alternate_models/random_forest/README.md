# Random Forest — Experiment Record

## Algorithm Family
Ensemble (bagging)

## Description
Averages 200 decision trees trained on bootstrap samples with random feature subsets. Reduces variance relative to a single tree and produces calibrated probability estimates.

## Configuration
```
{
  "n_estimators": 200,
  "min_samples_leaf": 5,
  "max_features": "sqrt",
  "n_jobs": -1,
  "random_state": 42
}
```

## How to Reproduce
```bash
python scripts/benchmark_classifiers.py random_forest --force
```

## Recorded Results

### Cross-Validation (5-fold GroupKFold, 80 training trips)
| Metric | Value |
|---|---|
| Macro F1 (mean) | 0.9275 |
| Top-1 Accuracy (mean) | 0.8774 |
| Log Loss (mean) | 0.0853 |

### Student-Testing Partition (10 trips)
| Metric | Value |
|---|---|
| Top-1 Accuracy | 0.8922 |
| Macro F1 | 0.9353 |
| Precision (macro) | 0.9412 |
| Recall (macro) | 0.9295 |
| Log Loss | 0.0659 |
| Route Continuity | 0.9551 |
| Candidate Miss Rate | 0.0010 |

### Timing
| | Value |
|---|---|
| Fit time (full train) | 46.9s |
| Inference | 2.19 ms / 1k rows |

## Notes
This model was not selected as the final model. See the comparison table in `results/models/comparison.csv` for the full ranking.
