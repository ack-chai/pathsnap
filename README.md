# PathSnap: Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning Hackathon (PES University, RR Campus).

## Structure
- `src/data/`      — parsing, inventory, projection, road-network graph
- `src/features/`  — candidate generation and feature engineering
- `src/models/`    — model training, comparison, Bayesian optimisation
- `src/decoder/`   — HMM + Viterbi route reconstruction
- `src/api/`       — FastAPI backend (Laptop 1)
- `ui/`            — evaluator UI (Laptop 2)
- `notebooks/`     — EDA
- `experiments/`   — logs, Optuna records, result CSVs
- `configs/`       — splits, hyperparameters, paths

## Quick start
```
pip install -r requirements.txt
python -m src.data.inventory data/raw --out experiments/inventory.csv
```

## Rules
- Fit scalers/encoders on training folds only.
- Evaluator trips: structural checks only, no label inspection.
- Ground truth stays server-side; never exposed to model or UI.
