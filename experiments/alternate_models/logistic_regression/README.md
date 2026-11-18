# Logistic Regression — Experiment Record

## Algorithm Family
Linear / probabilistic

## Description
Models the probability of a candidate being the correct road segment as a linear function of features. Fast and interpretable but limited to linear decision boundaries.

## Configuration
```
{
  "C": 1.0,
  "max_iter": 1000
}
```

## How to Reproduce
```bash
python scripts/benchmark_classifiers.py logistic_regression --force
```

## Recorded Results

### Cross-Validation (5-fold GroupKFold, 80 training trips)
| Metric | Value |
|---|---|
| Macro F1 (mean) | 0.9214 |
| Top-1 Accuracy (mean) | 0.8687 |
| Log Loss (mean) | 0.0961 |

### Student-Testing Partition (10 trips)
| Metric | Value |
|---|---|
| Top-1 Accuracy | 0.8810 |
| Macro F1 | 0.9277 |
| Precision (macro) | 0.9267 |
| Recall (macro) | 0.9287 |
| Log Loss | 0.0846 |
| Route Continuity | 0.9470 |
| Candidate Miss Rate | 0.0010 |

### Timing
| | Value |
|---|---|
| Fit time (full train) | 1.2s |
| Inference | 0.04 ms / 1k rows |

## Notes
This model was not selected as the final model. See the comparison table in `results/models/comparison.csv` for the full ranking.
