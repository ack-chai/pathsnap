# Random Forest comparison experiment

This is an archived comparison model, not the model used by the deployed
PathSnap pipeline. The deployed artifact remains the tuned XGBoost model in
`artifacts/model.joblib`.

## Model and reproducibility

The Random Forest implementation is part of the shared model registry in
`pathsnap/models.py`; it uses scikit-learn `RandomForestClassifier` with 200
trees, `min_samples_leaf=5`, `max_features="sqrt"`, and `random_state=42`.
The comparison runner trains it with the same candidate features and trip-level
cross-validation process as the other models:

```bash
python scripts/benchmark_classifiers.py random_forest --force
```

The experiment caps each fit at 600,000 candidate rows for runtime. Raw trip
data is not committed; follow the dataset setup instructions in the root
README before reproducing the run.

## Recorded results

These are the results from the recorded run in `results.json`:

| Metric | 5-fold training CV | Student Testing |
|---|---:|---:|
| Candidate-row accuracy | — | 97.36% |
| Macro F1 | 92.75% | 93.53% |
| Observation-level top-1 accuracy | 87.74% | 89.22% |
| Log loss | 0.0853 | 0.0659 |
| Route continuity | 94.85% | 95.51% |
| Candidate miss rate | — | 0.10% |

The full fold metrics, confusion matrix, timings, and model configuration are
in `results.json`. Candidate-row accuracy is not interchangeable with
observation-level top-1 accuracy: the latter selects the highest-scoring road
candidate for each GPS observation.
