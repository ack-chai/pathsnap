# Knn — Experiment Record

## Algorithm Family
Instance-based

## Description
Classifies a candidate by majority vote among its 25 nearest neighbours in feature space. No explicit training phase; inference cost is O(n). Requires feature scaling.

## Configuration
```
{
  "n_neighbors": 25,
  "weights": "distance",
  "n_jobs": -1
}
```

## How to Reproduce
```bash
python scripts/benchmark_classifiers.py knn --force
```

## Recorded Results

### Cross-Validation (5-fold GroupKFold, 80 training trips)
| Metric | Value |
|---|---|
| Macro F1 (mean) | 0.9153 |
| Top-1 Accuracy (mean) | 0.8611 |
| Log Loss (mean) | 0.1457 |

### Student-Testing Partition (10 trips)
| Metric | Value |
|---|---|
| Top-1 Accuracy | 0.8760 |
| Macro F1 | 0.9227 |
| Precision (macro) | 0.9202 |
| Recall (macro) | 0.9253 |
| Log Loss | 0.1171 |
| Route Continuity | 0.9402 |
| Candidate Miss Rate | 0.0010 |

### Timing
| | Value |
|---|---|
| Fit time (full train) | 0.0s |
| Inference | 75.05 ms / 1k rows |

## Notes
This model was not selected as the final model. See the comparison table in `results/models/comparison.csv` for the full ranking.
