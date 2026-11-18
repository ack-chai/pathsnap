# Decision Tree — Experiment Record

## Algorithm Family
Tree-based

## Description
Recursive binary partitioning of feature space. Fully interpretable and fast but prone to overfitting; max_depth=12 and min_samples_leaf=50 are used to regularise.

## Configuration
```
{
  "max_depth": 12,
  "min_samples_leaf": 50,
  "random_state": 42
}
```

## How to Reproduce
```bash
python scripts/benchmark_classifiers.py decision_tree --force
```

## Recorded Results

### Cross-Validation (5-fold GroupKFold, 80 training trips)
| Metric | Value |
|---|---|
| Macro F1 (mean) | 0.9122 |
| Top-1 Accuracy (mean) | 0.8698 |
| Log Loss (mean) | 0.1027 |

### Student-Testing Partition (10 trips)
| Metric | Value |
|---|---|
| Top-1 Accuracy | 0.8808 |
| Macro F1 | 0.9288 |
| Precision (macro) | 0.9352 |
| Recall (macro) | 0.9227 |
| Log Loss | 0.0744 |
| Route Continuity | 0.9503 |
| Candidate Miss Rate | 0.0010 |

### Timing
| | Value |
|---|---|
| Fit time (full train) | 13.8s |
| Inference | 0.04 ms / 1k rows |

## Notes
This model was not selected as the final model. See the comparison table in `results/models/comparison.csv` for the full ranking.
