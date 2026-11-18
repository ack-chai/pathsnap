# Gaussian Nb — Experiment Record

## Algorithm Family
Probabilistic (generative)

## Description
Assumes each feature is conditionally independent given the class label (Naive Bayes assumption). Very fast to train and produces probability estimates, but the independence assumption rarely holds perfectly.

## Configuration
```
{
  "var_smoothing": 1e-09
}
```

## How to Reproduce
```bash
python scripts/benchmark_classifiers.py gaussian_nb --force
```

## Recorded Results

### Cross-Validation (5-fold GroupKFold, 80 training trips)
| Metric | Value |
|---|---|
| Macro F1 (mean) | 0.8009 |
| Top-1 Accuracy (mean) | 0.8513 |
| Log Loss (mean) | 0.8983 |

### Student-Testing Partition (10 trips)
| Metric | Value |
|---|---|
| Top-1 Accuracy | 0.8689 |
| Macro F1 | 0.8030 |
| Precision (macro) | 0.7558 |
| Recall (macro) | 0.9156 |
| Log Loss | 0.8569 |
| Route Continuity | 0.9413 |
| Candidate Miss Rate | 0.0010 |

### Timing
| | Value |
|---|---|
| Fit time (full train) | 0.2s |
| Inference | 0.08 ms / 1k rows |

## Notes
This model was not selected as the final model. See the comparison table in `results/models/comparison.csv` for the full ranking.
