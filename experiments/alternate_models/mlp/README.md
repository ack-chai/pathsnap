# Mlp — Experiment Record

## Algorithm Family
Neural network

## Description
Two-layer fully-connected network (64→32 hidden units, ReLU activations) with early stopping. Can capture nonlinear interactions but requires feature scaling and more data to generalise well.

## Configuration
```
{
  "hidden_layer_sizes": "(64, 32)",
  "alpha": 0.0001,
  "batch_size": 512,
  "learning_rate_init": 0.001,
  "max_iter": 60,
  "early_stopping": true,
  "validation_fraction": 0.1,
  "n_iter_no_change": 6,
  "random_state": 42
}
```

## How to Reproduce
```bash
python scripts/benchmark_classifiers.py mlp --force
```

## Recorded Results

### Cross-Validation (5-fold GroupKFold, 80 training trips)
| Metric | Value |
|---|---|
| Macro F1 (mean) | 0.9112 |
| Top-1 Accuracy (mean) | 0.8710 |
| Log Loss (mean) | 0.0904 |

### Student-Testing Partition (10 trips)
| Metric | Value |
|---|---|
| Top-1 Accuracy | 0.8874 |
| Macro F1 | 0.9304 |
| Precision (macro) | 0.9402 |
| Recall (macro) | 0.9212 |
| Log Loss | 0.0681 |
| Route Continuity | 0.9562 |
| Candidate Miss Rate | 0.0010 |

### Timing
| | Value |
|---|---|
| Fit time (full train) | 9.1s |
| Inference | 0.10 ms / 1k rows |

## Notes
This model was not selected as the final model. See the comparison table in `results/models/comparison.csv` for the full ranking.
