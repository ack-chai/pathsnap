# Extra Trees — Experiment Record

## Algorithm Family
Ensemble (randomised trees)

## Description
Similar to Random Forest but splits are chosen at random rather than optimally, further reducing variance at the cost of slightly higher bias. Often faster to train.

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
python scripts/benchmark_classifiers.py extra_trees --force
```

## Recorded Results

### Cross-Validation (5-fold GroupKFold, 80 training trips)
| Metric | Value |
|---|---|
| Macro F1 (mean) | 0.9263 |
| Top-1 Accuracy (mean) | 0.8746 |
| Log Loss (mean) | 0.0833 |

### Student-Testing Partition (10 trips)
| Metric | Value |
|---|---|
| Top-1 Accuracy | 0.8877 |
| Macro F1 | 0.9309 |
| Precision (macro) | 0.9384 |
| Recall (macro) | 0.9238 |
| Log Loss | 0.0699 |
| Route Continuity | 0.9550 |
| Candidate Miss Rate | 0.0010 |

### Timing
| | Value |
|---|---|
| Fit time (full train) | 16.3s |
| Inference | 3.10 ms / 1k rows |

## Notes
This model was not selected as the final model. See the comparison table in `results/models/comparison.csv` for the full ranking.
