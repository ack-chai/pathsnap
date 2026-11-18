# Xgboost — Experiment Record

## Algorithm Family
Boosting

## Description
Gradient boosted trees with second-order gradient statistics and built-in regularisation (L1/L2). Selected as the final model after Bayesian hyperparameter optimisation.

## Configuration
```
{
  "n_estimators": 400,
  "learning_rate": 0.05,
  "max_depth": 7,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "tree_method": "hist",
  "random_state": 42,
  "n_jobs": -1,
  "eval_metric": "logloss"
}
```

## How to Reproduce
```bash
python scripts/benchmark_classifiers.py xgboost --force
```

## Recorded Results

### Cross-Validation (5-fold GroupKFold, 80 training trips)
| Metric | Value |
|---|---|
| Macro F1 (mean) | 0.9248 |
| Top-1 Accuracy (mean) | 0.8764 |
| Log Loss (mean) | 0.0810 |

### Student-Testing Partition (10 trips)
| Metric | Value |
|---|---|
| Top-1 Accuracy | 0.8983 |
| Macro F1 | 0.9367 |
| Precision (macro) | 0.9421 |
| Recall (macro) | 0.9315 |
| Log Loss | 0.0632 |
| Route Continuity | 0.9583 |
| Candidate Miss Rate | 0.0010 |

### Timing
| | Value |
|---|---|
| Fit time (full train) | 10.6s |
| Inference | 0.84 ms / 1k rows |

## Notes
This is the **final deployed model**, further tuned via Bayesian optimisation (`scripts/hyperparameter_search.py`).
