"""
Standalone training and evaluation script for: extra_trees
Algorithm family: Ensemble (randomised trees)

Usage (from repo root):
    python experiments/alternate_models/extra_trees/train_extra_trees.py

Outputs:
    experiments/alternate_models/extra_trees/results.json
    results/models/runs/extra_trees.json

Prerequisites:
    source .venv/bin/activate
    python scripts/extract_features.py   # builds cache/*.parquet
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score,
)
from sklearn.model_selection import GroupKFold

from sklearn.ensemble import ExtraTreesClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parents[3]
CACHE    = ROOT / "cache"
OUT_DIR  = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "results" / "models" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "dist_m", "bearing_diff", "road_len", "speed_mps", "accel",
    "heading_sin", "heading_cos", "road_sin", "road_cos",
    "lane_count", "is_tunnel", "is_bridge",
    "cand_rank", "n_candidates", "top2_gap", "prev_arc", "next_arc",
    "dist_prev", "dist_next", "dt_prev", "dt_next", "bearing_prev", "bearing_next",
]


def load_partition(part: str) -> pd.DataFrame:
    path = CACHE / f"{part}.parquet"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run scripts/extract_features.py first.")
    df = pd.read_parquet(path)
    cols = [c for c in FEATURES if c in df.columns]
    return df[["trip", "pidx", "arc", "y"] + cols].copy()


def pick_top1(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby(["trip", "pidx"])["p"].idxmax()
    return df.loc[idx]


def compute_metrics(df: pd.DataFrame) -> dict:
    y    = df["y"].values
    p    = df["p"].values
    yhat = (p >= 0.5).astype(int)
    top  = pick_top1(df)
    has_pos  = df.groupby(["trip", "pidx"])["y"].max()
    scored   = top[top.set_index(["trip", "pidx"]).index.map(has_pos) > 0]
    top1_acc = float(scored["y"].mean()) if len(scored) else float("nan")
    arc  = top["arc"].values
    cont = float((arc[1:] == arc[:-1]).mean()) if len(arc) > 1 else 1.0
    miss = float((has_pos == 0).mean())
    return dict(
        accuracy          = float(accuracy_score(y, yhat)),
        precision_macro   = float(precision_score(y, yhat, average="macro", zero_division=0)),
        recall_macro      = float(recall_score(y, yhat, average="macro", zero_division=0)),
        f1_macro          = float(f1_score(y, yhat, average="macro")),
        log_loss          = float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        top1_accuracy     = top1_acc,
        route_continuity  = cont,
        candidate_miss_rate = miss,
        confusion         = confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    )


def main():
    print(f"\n=== Training: extra_trees ===")
    train = load_partition("train")
    test  = load_partition("student_test")
    feat_cols = [c for c in FEATURES if c in train.columns]

    X_tr = train[feat_cols].values;  y_tr = train["y"].values
    X_te = test[feat_cols].values;   y_te = test["y"].values
    groups = train["trip"].values

    # -------------------------------------------------------------------
    # 5-fold GroupKFold cross-validation (groups = trip)
    # -------------------------------------------------------------------
    gkf = GroupKFold(n_splits=5)
    cv_f1, cv_top1, cv_ll = [], [], []
    print("Cross-validation:")
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_tr, y_tr, groups)):
        model = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=42)
        model.fit(X_tr[tr_idx], y_tr[tr_idx])
        p_val = model.predict_proba(X_tr[val_idx])[:, 1]
        val_df = train.iloc[val_idx].copy()
        val_df["p"] = p_val
        m = compute_metrics(val_df)
        cv_f1.append(m["f1_macro"]);  cv_top1.append(m["top1_accuracy"]);  cv_ll.append(m["log_loss"])
        print(f"  Fold {fold+1}: top1={m['top1_accuracy']:.4f}  f1={m['f1_macro']:.4f}  ll={m['log_loss']:.4f}")

    # -------------------------------------------------------------------
    # Refit on full training set, evaluate on student_test
    # -------------------------------------------------------------------
    print("\nFitting on full training set ...")
    t0 = time.time()
    model = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=42)
    model.fit(X_tr, y_tr)
    fit_sec = time.time() - t0

    t1 = time.time()
    p_te = model.predict_proba(X_te)[:, 1]
    infer_ms = (time.time() - t1) / len(X_te) * 1e6

    test_df = test.copy();  test_df["p"] = p_te
    test_m  = compute_metrics(test_df)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    result = dict(
        model            = "extra_trees",
        family           = "Ensemble (randomised trees)",
        params           = {
    "n_estimators": 200,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": 42
},
        cv_f1_macro_mean = float(np.mean(cv_f1)),
        cv_f1_macro_std  = float(np.std(cv_f1)),
        cv_top1_mean     = float(np.mean(cv_top1)),
        cv_log_loss_mean = float(np.mean(cv_ll)),
        fit_seconds_full       = fit_sec,
        test_infer_ms_per_1k_rows = infer_ms,
        test             = test_m,
    )
    (OUT_DIR  / "results.json").write_text(json.dumps(result, indent=2))
    (RUNS_DIR / "extra_trees.json").write_text(json.dumps(result, indent=2))

    print(f"\ncv_top1={np.mean(cv_top1):.4f}  test_top1={test_m['top1_accuracy']:.4f}  "
          f"test_f1={test_m['f1_macro']:.4f}  ll={test_m['log_loss']:.4f}")
    print(f"Results saved -> {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
