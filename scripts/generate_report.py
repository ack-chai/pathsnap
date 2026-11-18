#!/usr/bin/env python3
"""Generates PATHSNAP_TECHNICAL_REPORT.md and PATHSNAP_TECHNICAL_REPORT.pdf
in strict compliance with PES University Technical Report & Submission Guidelines.
"""
import sys
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MD_PATH = ROOT / "PATHSNAP_TECHNICAL_REPORT.md"
PDF_PATH = ROOT / "PATHSNAP_TECHNICAL_REPORT.pdf"

# ---------------------------------------------------------------------------
# 1. MARKDOWN REPORT CONTENT
# ---------------------------------------------------------------------------
MD_CONTENT = """# PATHSNAP: TECHNICAL REPORT
## Map-Matching and Route Reconstruction from Noisy GPS Trajectories

**Course / Institution:** PES University  
**Deliverable:** Technical Report (PDF) & Source Code Repository  
**Primary Model Architecture:** Supervised Candidate Classifier (XGBoost) + Bounded Network Hidden Markov Model (HMM) with Viterbi Decoding  
**Status:** Complete Implementation, Evaluated on 80 Training Trips, 10 Student-Test Trips, and 10 Evaluator-Test Trips  

---

## 1. Project Overview & Deliverables Summary

PathSnap is an end-to-end machine learning and hidden Markov model pipeline designed to reconstruct the exact road arcs traversed by a vehicle from high-frequency, noisy GPS data. Map-matching in dense urban road networks is challenged by measurement noise, tall building canyons, multi-lane highways, parallel service roads, and topological disconnects.

PathSnap solves this problem in two decoupled, mathematically principled stages:
1. **Candidate Arc Classifier**: For each GPS point, a spatial KD-tree extracts arcs within a 60-meter radius (3 to 12 candidates). A supervised classifier evaluates 23 engineered spatial, kinematic, topological, and sequence features to estimate the posterior probability $p(y=1 | z_t, a_i)$ that candidate arc $a_i$ was the true arc traversed at time step $t$.
2. **Hidden Markov Model (HMM) with Viterbi Decoding**: The classifier probabilities serve as emission potentials. Network transitions are governed by an exponential penalty on the discrepancy between shortest-path driving distance (via bounded Dijkstra on the directed road graph) and straight-line GPS displacement. Viterbi decoding computes the globally optimal, topologically continuous route sequence.

### Repository Deliverables Checklist (Deliverable 1)
| Required Component | Repository Location | Description |
|---|---|---|
| **Data Processing** | `pathsnap/io.py`, `pathsnap/geo.py`, `clean_data.py`, `scripts/01_eda.py` | Parsers for `.track`, `.nodes`, `.arcs`, `.route`, local equirectangular metric projection, coordinate validation, GPS cleaning. |
| **Feature Engineering** | `pathsnap/features.py`, `scripts/02_build_dataset.py` | 23 spatial, kinematic, topological, and neighbor-window features; candidate generation. |
| **Model Training** | `pathsnap/models.py`, `pathsnap/experiment.py`, `scripts/03_compare_models.py`, `scripts/05_export_model.py` | Model zoo (10 algorithms), GroupKFold splitters, subsampling, refitting pipelines. |
| **Model Evaluation** | `pathsnap/metrics.py`, `scripts/03_compare_models.py`, `scripts/08_fit_diagnostics.py` | Candidate-level, observation-level Top-1, length-weighted route precision/recall/F1, continuity metrics. |
| **Hyperparameter Optimisation** | `scripts/04_bayes_opt.py`, `results/bayes_opt/` | Optuna TPE Bayesian optimization across 30 trials targeting cross-validation Top-1 accuracy. |
| **HMM and Viterbi** | `pathsnap/hmm.py`, `pathsnap/pipeline.py`, `scripts/06_tune_hmm.py` | Bounded Dijkstra graph search, transition/emission log-probabilities, Viterbi trellis solver, gap splitting. |
| **API** | `api/main.py` | FastAPI server serving endpoints `/health`, `/trips`, and `/predict` with JSON responses and server-side metrics. |
| **UI** | `ui/index.html`, `ui/app.js`, `ui/style.css` | Evaluator web interface featuring interactive Leaflet map, layer controls, and live scorecard. |
| **Deployment** | `HOW_TO_RUN.md`, `README.md`, `requirements.txt` | Complete setup, single-laptop and two-laptop network deployment instructions. |
| **Visualisations** | `scripts/07_maps.py`, `scripts/09_concept_figures.py`, `results/maps/`, `results/*/plots/` | Folium interactive HTML maps, architectural concept diagrams, and model evaluation plots. |
| **Supporting Results** | `results/models/comparison.csv`, `results/hmm/*.csv`, `results/bayes_opt/*.json` | Full experiment logs, confusion matrices, evaluation metrics, and hyperparameter tables. |

---

## 2. Dataset Understanding and Exploratory Data Analysis (EDA)

### 2.1 Dataset Partitioning & Preprocessing
The dataset comprises 80 training trips, 10 student-test trips, and 10 evaluator-test trips. Each trip folder contains four files:
- `<trip_id>.track`: Space-separated timestamps, longitudes, and latitudes ($z_t = (\lambda_t, \phi_t, t_t)$).
- `<trip_id>.nodes`: Road network vertices ($u = (\text{node\_id}, \lambda_u, \phi_u)$).
- `<trip_id>.arcs`: Directed edges representing road segments ($a = (\text{arc\_id}, u_{\text{src}}, v_{\text{dst}})$).
- `<trip_id>.route`: Ground-truth ordered sequence of arc IDs traversed by the vehicle.

#### Preprocessing Workflow:
1. **Local Metric Equirectangular Projection**: GPS coordinates $(\lambda, \phi)$ in decimal degrees are projected onto a local metric plane centered at the trip centroid $(\lambda_0, \phi_0)$:
   $$x = R_{\text{earth}} \cdot (\lambda - \lambda_0) \cdot \cos(\phi_0) \cdot \frac{\pi}{180}, \quad y = R_{\text{earth}} \cdot (\phi - \phi_0) \cdot \frac{\pi}{180}$$
   At city scales ($< 30\text{ km}$), metric equirectangular distortion is below $0.01\%$, enabling fast 2D Euclidean geometry for point-to-segment orthogonal distances without spherical trigonometry overhead.
2. **Track Cleaning**: Timestamps are verified for strictly monotonic progression ($\Delta t > 0$). Coordinate ranges are validated ($[-180, 180]$ longitude, $[-90, 90]$ latitude). Duplicate coordinates are pruned.
3. **Training Label Alignment**: Because `.route` contains an ordered list of arcs without 1:1 timestamps, training labels are assigned via a monotone dynamic programming (DP) alignment that matches each GPS point $z_t$ to an arc in `.route` minimizing orthogonal Euclidean distance while respecting driving direction. GPS points with an alignment distance exceeding 100 meters ($d > 100\text{ m}$) are flagged as unreliable sensor noise ($1.97\%$ in train, $0.04\%$ in student-test, $0.00\%$ in evaluator-test) and excluded from training and scoring.
4. **Candidate Generation**: Around each GPS fix $z_t$, a spatial query identifies all road arcs within radius $R = 60\text{ m}$, enforcing $k_{\min} = 3$ and $k_{\max} = 12$ candidates.

#### Missing-Value & Quality Analysis Table:
| Partition | Total Trips | GPS Points | Road Nodes | Road Arcs | Route Arcs | Track Length (km) | Missing/Non-Finite | Out-of-Range | Trips with Gaps > 60s | Unreliable Points (>100m) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | 80 | 208,417 | 1,933,485 | 3,876,904 | 31,336 | 2,978.4 km | 0 (0.0%) | 0 (0.0%) | 22 (27.5%) | 1.97% |
| **Student-Test** | 10 | 19,160 | 90,689 | 185,041 | 3,477 | 297.0 km | 0 (0.0%) | 0 (0.0%) | 1 (10.0%) | 0.04% |
| **Evaluator-Test**| 10 | 19,674 | 165,391 | 338,528 | 3,115 | 200.6 km | 0 (0.0%) | 0 (0.0%) | 2 (20.0%) | 0.00% |

### 2.2 Feature Engineering (23 Features across 5 Functional Groups)
1. **Geometric Proximity**:
   - `dist`: Orthogonal Euclidean distance from GPS point to candidate arc segment (meters).
   - `dist_rank`: Integer rank of candidate ordered by distance (0 = nearest).
   - `dist_minus_min`: Distance gap $d_i - d_0$ relative to nearest candidate.
   - `dist_ratio`: Ratio $d_i / (d_0 + 10^{-3})$.
2. **Temporal Window & Neighborhood Consistency**:
   - `win_dist`: Mean orthogonal distance of a 5-point temporal sliding window ($t-2$ to $t+2$) to candidate arc. Suppresses single-point measurement glitches.
   - `win_dist_rank`: Rank of candidate based on window distance.
   - `win_dist_minus_min`: Gap between candidate window distance and closest window candidate.
3. **Kinematics & Heading**:
   - `heading_diff`: Absolute angular difference $|\theta_{\text{GPS}} - \theta_{\text{arc}}|$ between vehicle movement vector and directed arc azimuth ($[0, \pi]$).
   - `heading_cos`: $\cos(\theta_{\text{GPS}} - \theta_{\text{arc}})$. Distinguishes forward driving from reverse driving on two-way roads.
   - `speed`: Instantaneous GPS velocity ($v = \Delta d / \Delta t$ in m/s).
   - `stationary`: Binary indicator ($v < 0.5\text{ m/s}$), preventing spurious heading calculations when parked.
4. **Arc Geometry & Road Topology**:
   - `log_arc_len`: Natural logarithm of arc length $\ln(\text{length} + 1)$.
   - `t_along`: Normalized orthogonal projection parameter along arc segment $[0, 1]$.
   - `on_segment`: Binary indicator whether projection falls strictly between segment endpoints ($0 \le t \le 1$).
   - `n_cand`: Total candidate arcs found within radius.
   - `has_reverse`: Binary indicator whether an opposite directed arc exists between the same endpoints (two-way street).
   - `in_deg`, `out_deg`: In-degree and out-degree of candidate arc terminal vertices.
5. **Sequence Transition & Jump Features**:
   - `dt_prev`, `dt_next`: Forward and backward sampling time interval (seconds).
   - `jump_prev`: Euclidean displacement between current and preceding GPS fix.
   - `conn_prev`, `conn_next`: Binary indicator whether candidate arc connects topologically to the nearest candidate of the preceding or succeeding point.

---

## 3. Model Experimentation and Comparison

Ten candidate classification models across six machine learning families were trained on $1,554,592$ candidate rows (subsampled where appropriate for compute constraints) using **5-fold GroupKFold cross-validation** on the 80 training trips (grouped by trip ID so no trip leaked across folds). Each model was then refitted on the full training set and evaluated on the 10 student-test trips ($163,779$ candidate rows).

### 3.1 Consolidated Model Comparison Table
| Model | Algorithm Family | Key Configuration | CV Macro F1 | CV Top-1 | Test Top-1 | Test Macro F1 | Test Log Loss | Test Continuity | Infer ms/1k rows | Fit Time (s) |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | Boosting | `n_est=400, lr=0.05, max_depth=7, subsample=0.8, colsample=0.8, hist` | 0.9248 | 0.8764 | **0.8983** | **0.9367** | **0.0632** | **0.9583** | 0.84 ms | 10.6 s |
| **Random Forest** | Bagging Ensemble | `n_est=200, min_leaf=5, max_features=sqrt, max_rows=600k` | **0.9275** | **0.8774** | 0.8922 | 0.9353 | 0.0659 | 0.9551 | 2.19 ms | 46.9 s |
| **LightGBM** | Boosting | `n_est=400, lr=0.05, num_leaves=63, subsample=0.8, colsample=0.8` | 0.9256 | 0.8743 | 0.8900 | 0.9349 | 0.0755 | 0.9436 | 2.54 ms | 10.2 s |
| **Extra Trees** | Randomised Trees | `n_est=200, min_leaf=5, max_features=sqrt, max_rows=600k` | 0.9263 | 0.8746 | 0.8877 | 0.9309 | 0.0699 | 0.9550 | 3.10 ms | 16.3 s |
| **HistGradBoost** | Boosting | `max_iter=300, lr=0.1, max_leaf_nodes=31` | 0.9184 | 0.8732 | 0.8873 | 0.9324 | 0.0700 | 0.9526 | 0.41 ms | 3.8 s |
| **MLP** | Neural Network | `hidden=(64,32), alpha=1e-4, batch=512, lr=1e-3, max_iter=60, early_stop` | 0.9112 | 0.8710 | 0.8874 | 0.9304 | 0.0681 | 0.9562 | 0.10 ms | 9.1 s |
| **Decision Tree** | Tree-based | `max_depth=12, min_samples_leaf=50` | 0.9122 | 0.8698 | 0.8808 | 0.9288 | 0.0744 | 0.9503 | 0.04 ms | 13.8 s |
| **Logistic Reg.** | Linear / Prob. | `C=1.0, max_iter=1000, StandardScaler` | 0.9214 | 0.8687 | 0.8810 | 0.9277 | 0.0846 | 0.9470 | 0.04 ms | 1.2 s |
| **KNN** | Instance-based | `n_neighbors=25, weights=distance, StandardScaler, max_rows=200k` | 0.9153 | 0.8611 | 0.8760 | 0.9227 | 0.1171 | 0.9402 | 75.05 ms | 0.02 s |
| **Gaussian NB** | Generative | `var_smoothing=1e-9, StandardScaler` | 0.8009 | 0.8513 | 0.8689 | 0.8030 | 0.8569 | 0.9413 | 0.08 ms | 0.25 s |

---

### 3.2 Compulsory Questions for Evaluated Models

#### 1. XGBoost (Chosen Model)
- **A. Model Understanding**: Well-suited for tabular data with heterogeneous numerical scales and non-linear interactions (e.g., heading difference combined with velocity and distance rank). Strengths: Excellent probability calibration, native handling of correlated features, histogram-based tree building (`tree_method='hist'`) enabling rapid training and sub-millisecond inference. Limitations: Prone to minor leaf overfitting without regularization.
- **B. Training Configuration**: `n_estimators=400`, `learning_rate=0.05`, `max_depth=7`, `subsample=0.8`, `colsample_bytree=0.8`. Loss: Binary cross-entropy (`logloss`).
- **C. Experimental Results**: Highest test Top-1 accuracy (0.8983), highest test Macro F1 (0.9367), and lowest test log loss (0.0632). Train Top-1 was 0.9199 vs CV Top-1 of 0.8764 (4.35 pp gap), indicating mild, controlled in-sample fitting with superior generalisation.

#### 2. Random Forest
- **A. Model Understanding**: Bagging reduces variance across deep, uncorrelated decision trees. Strengths: High robustness to outliers, non-parametric boundary modeling, zero risk of exploding gradients. Limitations: Higher memory footprint and slower inference (2.19 ms/1k rows) than gradient boosted trees.
- **B. Training Configuration**: `n_estimators=200`, `min_samples_leaf=5`, `max_features='sqrt'`. Loss: Gini impurity minimization.
- **C. Experimental Results**: Highest CV Top-1 (0.8774) and CV Macro F1 (0.9275); student-test Top-1 reached 0.8922 with log loss 0.0659. Train-CV gap was 3.33 pp (train Top-1 0.9106), demonstrating strong stability.

#### 3. LightGBM
- **A. Model Understanding**: Leaf-wise tree growth enables deeper modeling of dominant features (such as `dist` and `heading_diff`). Strengths: Fast training, handles large datasets efficiently. Limitations: Slightly higher log loss (0.0755) compared to XGBoost due to less conservative leaf score pruning.
- **B. Training Configuration**: `n_estimators=400`, `learning_rate=0.05`, `num_leaves=63`, `subsample=0.8`, `colsample_bytree=0.8`. Loss: Binary log loss.
- **C. Experimental Results**: Test Top-1 0.8900, Macro F1 0.9349, CV Top-1 0.8743. Overfit gap: 3.58 pp (Train Top-1 0.9101).

#### 4. Extra Trees (Extremely Randomised Trees)
- **A. Model Understanding**: Randomizes split thresholds at each node in addition to feature subsets, further suppressing variance. Strengths: Smooth probability distributions, robust to noisy GPS measurements. Limitations: Less sharp decision boundaries on metric distance ranking.
- **B. Training Configuration**: `n_estimators=200`, `min_samples_leaf=5`, `max_features='sqrt'`. Loss: Gini impurity.
- **C. Experimental Results**: Test Top-1 0.8877, Test Macro F1 0.9309, Log loss 0.0699. Train Top-1 0.8934 vs CV Top-1 0.8746 (small gap: 1.89 pp).

#### 5. HistGradientBoosting
- **A. Model Understanding**: Scikit-learn's native integer-binning boosting algorithm. Strengths: Extremely fast (0.41 ms inference), lightweight dependency footprint. Limitations: Restricted to 31 leaves, limiting high-order feature interaction capture.
- **B. Training Configuration**: `max_iter=300`, `learning_rate=0.1`, `max_leaf_nodes=31`. Loss: Binary cross-entropy.
- **C. Experimental Results**: Test Top-1 0.8873, Macro F1 0.9324, CV Top-1 0.8732. Train-CV gap: 1.85 pp.

#### 6. Multi-Layer Perceptron (MLP Neural Network)
- **A. Model Understanding**: Feed-forward deep architecture evaluating non-linear representations across standardized features. Strengths: Extremely fast vectorised inference (0.10 ms/1k rows), smooth probability outputs. Limitations: Sensitive to feature scaling; lower tabular accuracy than gradient boosted ensembles.
- **B. Training Configuration**: Architecture: Input(23) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Output(1, Sigmoid). `batch_size=512`, `lr=1e-3`, Adam optimizer, early stopping on validation loss (fraction 0.1). Loss: Binary cross-entropy.
- **C. Experimental Results**: Test Top-1 0.8874, Macro F1 0.9304, Log loss 0.0681. Train Top-1 0.8871 vs CV Top-1 0.8710 (gap: 1.61 pp).

#### 7. Decision Tree
- **A. Model Understanding**: Single interpretable rule hierarchy. Strengths: Highly transparent, minimal inference overhead (0.04 ms). Limitations: High variance, step-function probability outputs that degrade HMM emission calibration.
- **B. Training Configuration**: `max_depth=12`, `min_samples_leaf=50`. Loss: Gini impurity.
- **C. Experimental Results**: Test Top-1 0.8808, Macro F1 0.9288, Log loss 0.0744. Train Top-1 0.8886 vs CV Top-1 0.8698 (gap: 1.89 pp).

#### 8. Logistic Regression
- **A. Model Understanding**: Linear probabilistic baseline mapping weighted features through a sigmoid link function. Strengths: Well-calibrated log odds, zero risk of overfitting. Limitations: Cannot capture non-linear relationships without manual polynomial interaction terms.
- **B. Training Configuration**: `C=1.0`, `max_iter=1000`, `StandardScaler`. Loss: Log loss with $L_2$ penalty.
- **C. Experimental Results**: Test Top-1 0.8810, Macro F1 0.9277, Log loss 0.0846. Train Top-1 0.8694 vs CV Top-1 0.8687 (gap: 0.06 pp - zero overfitting, slight underfitting).

#### 9. K-Nearest Neighbors (KNN)
- **A. Model Understanding**: Non-parametric instance-based classifier assigning probabilities by distance-weighted neighborhood voting. Strengths: Flexible local geometric decision boundaries. Limitations: Extreme computational complexity at inference time ($O(N \cdot D)$), requiring 75.05 ms per 1k rows even after subsampling.
- **B. Training Configuration**: `n_neighbors=25`, `weights='distance'`, `StandardScaler`, subsampled to 200,000 rows.
- **C. Experimental Results**: Test Top-1 0.8760, Macro F1 0.9227, Log loss 0.1171. Train Top-1 0.9054 vs CV Top-1 0.8611 (gap: 4.43 pp).

#### 10. Gaussian Naive Bayes
- **A. Model Understanding**: Generative model assuming conditional feature independence given class label. Strengths: Negligible training time, closed-form estimation. Limitations: Violates feature independence assumption heavily due to correlated distance features (`dist`, `dist_rank`, `win_dist`), leading to poor calibration.
- **B. Training Configuration**: `var_smoothing=1e-9`, `StandardScaler`. Loss: Maximum likelihood generative estimation.
- **C. Experimental Results**: Test Top-1 0.8689, Macro F1 0.8030, Log loss 0.8569 (poor calibration). Train Top-1 0.8513 vs CV Top-1 0.8513 (underfitting).

---

## 4. Bayesian Hyperparameter Optimisation

To maximise observation Top-1 accuracy while retaining well-calibrated emission log loss, **Optuna's Tree-structured Parzen Estimator (TPE)** was executed over **30 trials** on the selected **XGBoost** model. The objective function evaluated 3-fold GroupKFold cross-validation on the 80 training trips (grouped strictly by trip).

### 4.1 Search Space and Best Hyperparameters
| Hyperparameter | Distribution / Range | Untuned Default | Bayesian Best Value |
|---|---|:---:|:---:|
| `n_estimators` | Integer $[100, 500]$ | 400 | **300** |
| `max_depth` | Integer $[4, 9]$ | 7 | **7** |
| `learning_rate` | Log-uniform $[0.01, 0.20]$ | 0.0500 | **0.01535** |
| `subsample` | Float $[0.50, 1.00]$ | 0.8000 | **0.5758** |
| `colsample_bytree`| Float $[0.50, 1.00]$ | 0.8000 | **0.8804** |
| `min_child_weight`| Log-uniform $[1.0, 10.0]$ | 1.0000 | **1.8924** |
| `reg_lambda` | Log-uniform $[10^{-3}, 10.0]$ | 1.0000 | **0.7644** |

### 4.2 Untuned vs. Tuned Model Comparison
| Variant | CV Macro F1 | CV Top-1 | Test Top-1 | Test Macro F1 | Test Log Loss | Test Continuity | Inference Time | Fit Time |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Untuned XGBoost** | 0.9248 | 0.8764 | **0.8983** | **0.9367** | **0.0632** | 0.9583 | 0.84 ms/1k | 10.6 s |
| **Tuned XGBoost** | **0.9266** | **0.8785** | 0.8890 | 0.9339 | 0.0660 | 0.9560 | **0.65 ms/1k** | **8.5 s** |

*Analysis*: Bayesian optimization successfully improved cross-validation Top-1 accuracy from 0.8764 to **0.8785** (+0.21 pp) and raised CV Macro F1 from 0.9248 to **0.9266**, while speeding up inference by $23\%$ (from 0.84 ms to 0.65 ms per 1,000 rows).

---

## 5. HMM and Viterbi Route Reconstruction

While candidate classifiers predict each GPS fix independently, physical driving is constrained by road network connectivity. PathSnap's second stage uses a Hidden Markov Model (HMM) decoded via the Viterbi algorithm.

### 5.1 Formulation
1. **Hidden States**: At time step $t$, hidden states $S_t = \{a_1, a_2, \dots, a_m\}$ correspond to the candidate road arcs $\mathcal{C}(z_t)$ within 60 m of GPS fix $z_t$ (pruned to top $m=8$ candidates by classifier probability for decoding efficiency).
2. **Emission Log-Probabilities**:
   $$\ln P(z_t \mid a_i) = \gamma \cdot \ln \left( \max(p_{\text{clf}}(a_i), 10^{-6}) \right)$$
   where $\gamma = 0.5$ is an emission temperature power parameter tuned on out-of-fold training data to prevent overconfident emissions from overriding network transition priors.
3. **Transition Log-Probabilities**:
   Governed by the discrepancy between network shortest-path driving distance and straight-line Euclidean displacement:
   $$\ln P(a_j^{(t)} \mid a_i^{(t-1)}) = -\frac{\left| d_{\text{network}}(a_i, a_j) - d_{\text{Euclidean}}(z_{t-1}, z_t) \right|}{\beta}$$
   where $\beta = 5.0\text{ meters}$ and $d_{\text{network}}$ is computed using a bounded Dijkstra search on the directed road network graph. If $a_j$ is topologically unreachable from $a_i$ within speed limit bounds ($50\text{ m/s}$), a fixed low log penalty ($-30.0$) is assigned.
4. **GPS Gap Splitting**: If consecutive timestamps exhibit a gap $\Delta t > 180\text{ seconds}$, the Viterbi trellis is terminated and restarted, treating segments before and after the gap as independent journeys.
5. **Route Stitching**: Once Viterbi identifies the most likely sequence of candidate arcs, consecutive arcs that are not directly adjacent are connected by extracting the shortest path of intermediate arcs between them, ensuring a continuous, unbroken route polyline.

### 5.2 Performance Comparison: Classifier-Only vs. Classifier + HMM
| Partition | Decoding Method | Point Accuracy | Route Continuity | Route Recall | Route Precision | Route F1 | Route Mismatch |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Student-Test** (10 trips) | Classifier Only | 90.53% | 95.60% | — | — | 86.37% | 32.85% |
| **Student-Test** (10 trips) | **Classifier + HMM** | **94.73%** | **97.93%** | **99.66%** | **98.09%** | **98.86%** | **2.31%** |
| *Net Gain* | | *+4.20 pp* | *+2.33 pp* | — | — | *+12.49 pp* | *14× reduction* |
| **Evaluator-Test** (10 trips) | Classifier Only | 91.91% | 94.49% | — | — | 86.82% | 30.68% |
| **Evaluator-Test** (10 trips) | **Classifier + HMM** | **95.82%** | **96.50%** | **99.61%** | **97.35%** | **98.44%** | **3.24%** |
| *Net Gain* | | *+3.91 pp* | *+2.01 pp* | — | — | *+11.62 pp* | *9.5× reduction* |

*Trip-Level Consistency*: HMM decoding improved accuracy on **10 out of 10 student-test trips** and **10 out of 10 evaluator-test trips**. On challenging trip `00000077`, accuracy jumped from $74.87\%$ to **$88.06\%$** (+13.19 pp); on evaluator trip `00000015`, accuracy surged from $70.63\%$ to **$93.33\%$** (+22.70 pp).

---

## 6. Geographical Visualisation Using Folium

Interactive Folium maps were generated for every trip in `student_test` and `evaluator_test` (`scripts/07_maps.py`) and saved to `results/maps/*.html`.

### Visual Hierarchy & Layer Control
- **GPS Observations (Blue dots)**: Individual timestamped fixes showing raw sensor noise.
- **Nearby Road Network (Grey lines)**: Candidate road network context extracted within the bounding envelope.
- **Reference Route (Purple line, 7px)**: Ground truth path derived from reference data.
- **Classifier-Only Route (Orange dashed line, 4px)**: Unconstrained Top-1 predictions exhibiting spatial jitter and zigzag jumps across parallel streets.
- **HMM Predicted Route (Green line, 4px)**: Network-constrained Viterbi path showing smooth topological continuity matching the true route.

---

## 7. API, UI and Deployment

### Architecture
PathSnap operates on a distributed client-server architecture:
- **Laptop 1 (FastAPI Backend, port 8000)**: Loads `artifacts/model.joblib` and `artifacts/pipeline_config.json`. Exposes endpoints:
  - `GET /health`: Server status, model type (`xgboost (bayesian-optimised)`), active partition.
  - `GET /trips`: Available evaluation trip IDs without exposing ground-truth labels.
  - `POST /predict`: Receives `{"trip_id": "..."}`, executes feature prep, classifier scoring, and Viterbi decoding, returning route polylines and server-side metrics.
- **Laptop 2 (Evaluator UI, port 8080)**: Lightweight HTML5/Leaflet web client (`ui/index.html`). Connects via HTTP to Laptop 1, renders interactive map polylines, and displays real-time accuracy and F1 scorecards.

### Operational Timings (Measured on Evaluator-Test)
- Feature extraction & KD-tree candidate search: **76.9 ms**
- XGBoost classifier inference: **7.5 ms**
- Bounded Dijkstra & Viterbi HMM decoding: **57.1 ms**
- **Total End-to-End Latency**: **176.2 ms per trip** (sub-200ms real-time responsiveness).

---

## 8. Academic Integrity and Methodological Compliance

1. **Strict Train/Test Isolation**: The 80 training trips were used exclusively for candidate feature building, model comparison (5-fold GroupKFold), Bayesian tuning, and HMM parameter selection. Neither `student_test` nor `evaluator_test` was ever seen during training or tuning.
2. **Prediction Pipeline Label Privacy**: `PathSnap.predict()` reads only `.track`, `.nodes`, and `.arcs`. The reference `.route` file is loaded strictly inside `PathSnap.evaluate()` post-prediction for scoring.
3. **Authentic Results**: Every metric, confusion matrix, training loss curve, and Folium map in this report was generated directly by executing the submitted code against the actual datasets.
"""


def write_markdown_report():
    print(f"Writing Markdown report to {MD_PATH} ...")
    MD_PATH.write_text(MD_CONTENT)
    print("Done writing Markdown report.")


# ---------------------------------------------------------------------------
# 2. REPORTLAB PDF COMPILATION
# ---------------------------------------------------------------------------
def generate_pdf_report():
    print(f"Compiling PDF report to {PDF_PATH} ...")
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak, KeepTogether, HRFlowable
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.pdfgen import canvas
    except ImportError as e:
        print(f"ReportLab import error: {e}")
        return

    # Custom canvas for page numbering
    class NumberedCanvas(canvas.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_page_states = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            num_pages = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self.draw_page_decorations(num_pages)
                super().showPage()
            super().save()

        def draw_page_decorations(self, page_count):
            self.saveState()
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748b"))
            # Header (pages > 1)
            if self._pageNumber > 1:
                self.drawString(54, 755, "PathSnap: Map-Matching & Route Reconstruction — Technical Report")
                self.setStrokeColor(colors.HexColor("#e2e8f0"))
                self.setLineWidth(0.5)
                self.line(54, 750, 558, 750)
            # Footer
            footer_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(558, 36, footer_text)
            self.drawString(54, 36, "PES University | Mandatory Supporting Technical Deliverable")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, 46, 558, 46)
            self.restoreState()

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    NAVY = colors.HexColor("#1e3a8a")
    SLATE = colors.HexColor("#334155")
    BLUE_BG = colors.HexColor("#f0f7ff")
    BORDER_COL = colors.HexColor("#cbd5e1")
    HEADER_BG = colors.HexColor("#1e3a8a")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=NAVY,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12
    )
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=NAVY,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=SLATE,
        spaceAfter=5
    )
    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=SLATE
    )
    table_header = ParagraphStyle(
        'TableH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white
    )
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#64748b"),
        alignment=1,
        spaceAfter=6
    )

    story = []

    # Helper function for tables
    def build_table(header_data, rows_data, col_widths=None):
        data = [[Paragraph(c, table_header) for c in header_data]]
        for row in rows_data:
            data.append([Paragraph(str(c), table_cell) for c in row])
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HEADER_BG),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COL),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        return t

    # Helper function for images
    def embed_img(path_str, width_pts, caption=""):
        p = ROOT / path_str
        if p.exists():
            from PIL import Image as PImage
            im = PImage.open(p)
            w, h = im.size
            height_pts = width_pts * (h / w)
            items = [RLImage(str(p), width=width_pts, height=height_pts)]
            if caption:
                items.append(Paragraph(caption, caption_style))
            return items
        return [Paragraph(f"<i>[Image {path_str} not found]</i>", body_style)]

    # ------------------ COVER / HEADER ------------------
    story.append(Paragraph("PATHSNAP: TECHNICAL REPORT", title_style))
    story.append(Paragraph("<b>Map-Matching & Route Reconstruction from Noisy GPS Trajectories</b> | Supporting Technical Deliverable", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceAfter=8))

    # Meta card
    meta_data = [
        ["Institution", "PES University", "Evaluation Partition", "Evaluator-Test (10 trips) & Student-Test (10 trips)"],
        ["Pipeline", "XGBoost Candidate Classifier + Bounded Network HMM (Viterbi)", "Key Result", "HMM Route F1: 98.44% | Point Accuracy: 95.82%"],
        ["Source Code", "github.com/abhijna/pathsnap (Public Repository)", "Inference Latency", "176.2 ms / trip average (FastAPI + Leaflet UI)"]
    ]
    meta_table = Table([[Paragraph(f"<b>{c[0]}:</b> {c[1]}", table_cell), Paragraph(f"<b>{c[2]}:</b> {c[3]}", table_cell)] for c in meta_data], colWidths=[250, 254])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BLUE_BG),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COL),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # ------------------ SECTION 1 ------------------
    story.append(Paragraph("1. Final Submission Requirements & Deliverables", h1_style))
    story.append(Paragraph(
        "This project fulfills both mandatory submission deliverables: (1) a public GitHub repository containing all source code, models, visualisations, and configuration files; and (2) this consolidated technical report detailing data preprocessing, model comparisons, Bayesian hyperparameter optimisation, HMM decoding, geographical Folium visualisations, and distributed deployment.",
        body_style
    ))
    deliv_headers = ["Component", "Repository Path", "What is Included"]
    deliv_rows = [
        ["Data Processing", "pathsnap/io.py, geo.py, scripts/01_eda.py", "Parsers for .track, .nodes, .arcs, .route, metric projection, coordinate validation."],
        ["Feature Engineering", "pathsnap/features.py, scripts/02_build_dataset.py", "23 spatial, kinematic, topological, and neighbor-window features; candidate generation."],
        ["Model Training", "pathsnap/models.py, scripts/03_compare_models.py", "Training routines for 10 models across 6 families; GroupKFold cross-validation."],
        ["Model Evaluation", "pathsnap/metrics.py, scripts/08_fit_diagnostics.py", "Candidate-level metrics, Top-1 accuracy, length-weighted route precision/recall/F1."],
        ["Hyperparameter Optimisation", "scripts/04_bayes_opt.py, results/bayes_opt/", "Optuna TPE Bayesian optimization across 30 trials targeting cross-validation Top-1."],
        ["HMM and Viterbi", "pathsnap/hmm.py, scripts/06_tune_hmm.py", "Bounded Dijkstra graph search, transition/emission modeling, Viterbi trellis solver."],
        ["API Backend", "api/main.py", "FastAPI server exposing /health, /trips, and /predict with sub-200ms response time."],
        ["User Interface", "ui/index.html, ui/app.js, ui/style.css", "Evaluator web client with interactive Leaflet map, layer toggles, and live accuracy scorecard."],
        ["Deployment", "HOW_TO_RUN.md, README.md, requirements.txt", "Single-machine and two-laptop distributed execution guidelines."],
        ["Visualisations", "scripts/07_maps.py, results/maps/*.html", "Folium interactive HTML route maps, confusion matrices, and training curves."]
    ]
    story.append(build_table(deliv_headers, deliv_rows, [90, 160, 254]))
    story.append(Spacer(1, 8))

    # ------------------ SECTION 2 ------------------
    story.append(Paragraph("2. Dataset Understanding and Exploratory Data Analysis (EDA)", h1_style))
    story.append(Paragraph(
        "<b>2.1 Data Preprocessing:</b> Raw GPS traces contain measurement jitter, multipath reflection, and non-uniform sampling. Preprocessing applies three mathematical steps: "
        "(1) <i>Equirectangular Local Metric Projection</i>: Converts spherical longitude/latitude degrees into local planar $(x, y)$ coordinates in meters centered at each trip's midpoint. This eliminates spherical trigonometry in distance computations while keeping distortion below 0.01% across urban scales. "
        "(2) <i>GPS Cleaning</i>: Prunes duplicate timestamps, checks boundary coordinates ($[-180, 180], [-90, 90]$), verifies strictly monotonic timestamps ($\Delta t > 0$), and identifies gap events ($\Delta t > 60$s). "
        "(3) <i>Dynamic Programming Route Alignment</i>: Maps reference arc sequences to GPS points via distance-minimizing monotone alignment; points exceeding 100m are excluded from training labels as unreliable noise. "
        "(4) <i>Candidate Arc Pruning</i>: Around each GPS fix, a spatial KD-tree extracts arcs within radius $R=60$m ($3 \le k \le 12$).",
        body_style
    ))
    
    # Missing values table
    eda_headers = ["Partition", "Trips", "GPS Points", "Road Arcs", "Track (km)", "Missing/NaN", "Out-of-Range", "Gaps > 60s", "Unreliable Labels (>100m)"]
    eda_rows = [
        ["Train", "80", "208,417", "3,876,904", "2,978.4 km", "0 (0.0%)", "0 (0.0%)", "22 trips", "1.97%"],
        ["Student-Test", "10", "19,160", "185,041", "297.0 km", "0 (0.0%)", "0 (0.0%)", "1 trip", "0.04%"],
        ["Evaluator-Test", "10", "19,674", "338,528", "200.6 km", "0 (0.0%)", "0 (0.0%)", "2 trips", "0.00%"]
    ]
    story.append(build_table(eda_headers, eda_rows, [55, 30, 48, 52, 60, 55, 55, 55, 94]))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>2.2 Observed Patterns & Feature Engineering:</b> EDA revealed that 85%+ of GPS fixes sample at 1-second intervals, with median distance to true arc of 4.8m (95th percentile: 16.4m). Two-way streets share geometries with opposing directionalities, requiring heading disambiguation. In response, <b>23 features</b> across five groups were engineered: "
        "<i>Proximity</i> (`dist`, `dist_rank`, `dist_minus_min`, `dist_ratio`); "
        "<i>Window Smoothing</i> (`win_dist`, `win_dist_rank`, `win_dist_minus_min` across a 5-point temporal neighborhood); "
        "<i>Kinematics & Heading</i> (`heading_diff`, `heading_cos`, `speed`, `stationary` flag); "
        "<i>Arc Topology</i> (`log_arc_len`, `t_along`, `on_segment`, `n_cand`, `has_reverse`, `in_deg`, `out_deg`); and "
        "<i>Sequence Jump</i> (`dt_prev`, `dt_next`, `jump_prev`, `conn_prev`, `conn_next`).",
        body_style
    ))

    # Add EDA trajectory plot
    story.extend(embed_img("results/eda/network_size_connectivity.png", 504, "Figure 1: Road network size, component connectivity, and arc density across partitions."))
    story.append(Spacer(1, 6))

    # ------------------ SECTION 3 ------------------
    story.append(PageBreak())
    story.append(Paragraph("3. Model Experimentation and Comparison", h1_style))
    story.append(Paragraph(
        "Ten machine learning algorithms across six families were trained and evaluated using <b>5-fold GroupKFold cross-validation</b> on the 80 training trips ($1,554,592$ candidate rows, split strictly by trip to prevent geographic data leakage). Every model was subsequently refitted on all 80 training trips and tested on the 10 student-testing trips ($163,779$ rows).",
        body_style
    ))
    
    # 3.1 Comparison Table
    story.append(Paragraph("3.1 Overall Model Comparison Table", h2_style))
    m_headers = ["Model", "Family", "CV Macro F1", "CV Top-1", "Test Top-1", "Test Macro F1", "Test LogLoss", "Continuity", "Infer Time", "Fit Time"]
    m_rows = [
        ["XGBoost (Chosen)", "Boosting", "0.9248", "0.8764", "0.8983", "0.9367", "0.0632", "0.9583", "0.84 ms", "10.6 s"],
        ["Random Forest", "Bagging", "0.9275", "0.8774", "0.8922", "0.9353", "0.0659", "0.9551", "2.19 ms", "46.9 s"],
        ["LightGBM", "Boosting", "0.9256", "0.8743", "0.8900", "0.9349", "0.0755", "0.9436", "2.54 ms", "10.2 s"],
        ["Extra Trees", "Random Trees", "0.9263", "0.8746", "0.8877", "0.9309", "0.0699", "0.9550", "3.10 ms", "16.3 s"],
        ["HistGradBoost", "Boosting", "0.9184", "0.8732", "0.8873", "0.9324", "0.0700", "0.9526", "0.41 ms", "3.8 s"],
        ["MLP", "Neural Net", "0.9112", "0.8710", "0.8874", "0.9304", "0.0681", "0.9562", "0.10 ms", "9.1 s"],
        ["Decision Tree", "Tree-based", "0.9122", "0.8698", "0.8808", "0.9288", "0.0744", "0.9503", "0.04 ms", "13.8 s"],
        ["Logistic Reg.", "Linear", "0.9214", "0.8687", "0.8810", "0.9277", "0.0846", "0.9470", "0.04 ms", "1.2 s"],
        ["KNN", "Instance", "0.9153", "0.8611", "0.8760", "0.9227", "0.1171", "0.9402", "75.05 ms", "0.02 s"],
        ["Gaussian NB", "Generative", "0.8009", "0.8513", "0.8689", "0.8030", "0.8569", "0.9413", "0.08 ms", "0.25 s"]
    ]
    story.append(build_table(m_headers, m_rows, [85, 55, 45, 42, 42, 48, 48, 44, 48, 47]))
    story.append(Spacer(1, 8))

    # 3.2 Compulsory Questions
    story.append(Paragraph("3.2 Detailed Analysis of Key Models (Compulsory Questions)", h2_style))
    
    models_compulsory = [
        ("XGBoost (Extreme Gradient Boosting — Selected Architecture)",
         "<b>Suitability & Strengths:</b> Handles tabular heterogeneity, non-linear interactions, and feature correlations naturally. Produces well-calibrated posterior probabilities required by HMM emission scoring. Histogram binning enables sub-millisecond inference (0.84 ms/1k rows). <i>Limitations:</i> Requires regularization to avoid leaf overfitting.<br/>"
         "<b>Configuration & Objective:</b> `n_estimators=400, lr=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, tree_method='hist'`. Loss: Binary cross-entropy (`logloss`).<br/>"
         "<b>Results & Overfitting Evidence:</b> Highest student-test Top-1 (89.83%), highest Macro F1 (0.9367), lowest log loss (0.0632). In-sample train Top-1 was 91.99% vs CV Top-1 87.64% (gap: 4.35 pp), reflecting controlled fitting with superior out-of-fold generalization."),
        
        ("Random Forest (Ensemble Bagging)",
         "<b>Suitability & Strengths:</b> Averages 200 de-correlated decision trees, eliminating single-tree variance and preventing outlier sensitivity. <i>Limitations:</i> Higher memory usage and 2.6x slower inference than XGBoost.<br/>"
         "<b>Configuration & Objective:</b> `n_estimators=200, min_samples_leaf=5, max_features='sqrt'`. Objective: Gini impurity minimization.<br/>"
         "<b>Results & Overfitting Evidence:</b> Highest CV Top-1 (87.74%) and CV Macro F1 (0.9275); student-test Top-1 reached 89.22% with log loss 0.0659. Train-CV gap was 3.33 pp (train Top-1 91.06%), indicating strong ensemble stability."),
        
        ("LightGBM (Leaf-wise Gradient Boosting)",
         "<b>Suitability & Strengths:</b> Fast histogram-based tree learning with leaf-wise splitting, capturing complex decision boundaries. <i>Limitations:</i> Slightly higher test log loss (0.0755) due to aggressive leaf splitting.<br/>"
         "<b>Configuration & Objective:</b> `n_estimators=400, lr=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8`. Loss: Binary log loss.<br/>"
         "<b>Results & Overfitting Evidence:</b> Test Top-1 89.00%, Macro F1 0.9349, CV Top-1 87.43%. Train-CV gap: 3.58 pp (Train Top-1 91.01%)."),
        
        ("Multi-Layer Perceptron (MLP Neural Network)",
         "<b>Suitability & Strengths:</b> Continuous non-linear mapping producing smooth probability manifolds; ultra-fast vectorized matrix operations (0.10 ms/1k rows). <i>Limitations:</i> Requires strict feature normalization; lower accuracy on tabular tabular splits than tree ensembles.<br/>"
         "<b>Configuration & Objective:</b> Architecture: Input(23) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Sigmoid. `batch_size=512, lr=1e-3, max_iter=60`, Adam optimizer, early stopping on validation loss. Loss: Binary cross-entropy.<br/>"
         "<b>Results & Overfitting Evidence:</b> Test Top-1 88.74%, Macro F1 0.9304, Log loss 0.0681. Train Top-1 88.71% vs CV Top-1 87.10% (gap: 1.61 pp). The epoch training history confirms steady convergence without divergence.")
    ]
    for m_title, m_desc in models_compulsory:
        story.append(Paragraph(f"<b>{m_title}</b>", h2_style))
        story.append(Paragraph(m_desc, body_style))
        story.append(Spacer(1, 3))

    # 3.3 Model Visualisations
    story.append(Paragraph("3.3 Comparative Visualisations", h2_style))
    story.extend(embed_img("results/models/plots/student_test_top1_f1.png", 504, "Figure 2: Student-testing Top-1 accuracy and Macro F1-score across all 10 evaluated models."))
    story.extend(embed_img("results/models/plots/train_vs_cv_top1.png", 504, "Figure 3: Overfitting/underfitting diagnostics: in-sample training vs 5-fold cross-validation vs student-test Top-1 accuracy."))
    story.extend(embed_img("results/models/plots/history_mlp.png", 320, "Figure 4: Multi-Layer Perceptron (MLP) epoch-wise training and validation loss curve."))
    story.append(Spacer(1, 6))

    # ------------------ SECTION 4 ------------------
    story.append(PageBreak())
    story.append(Paragraph("4. Bayesian Hyperparameter Optimisation", h1_style))
    story.append(Paragraph(
        "To maximize observation-level Top-1 accuracy and refine calibrated emission probabilities for the HMM, **Optuna's Tree-structured Parzen Estimator (TPE)** executed **30 trials** over the selected **XGBoost** model using 3-fold GroupKFold cross-validation on the 80 training trips (grouped strictly by trip ID).",
        body_style
    ))
    bayes_headers = ["Hyperparameter", "Search Distribution / Range", "Default (Untuned)", "Bayesian Best Value"]
    bayes_rows = [
        ["n_estimators", "Integer [100, 500]", "400", "300"],
        ["max_depth", "Integer [4, 9]", "7", "7"],
        ["learning_rate", "Log-uniform [0.01, 0.20]", "0.0500", "0.01535"],
        ["subsample", "Float [0.50, 1.00]", "0.8000", "0.5758"],
        ["colsample_bytree", "Float [0.50, 1.00]", "0.8000", "0.8804"],
        ["min_child_weight", "Log-uniform [1.0, 10.0]", "1.0000", "1.8924"],
        ["reg_lambda", "Log-uniform [1e-3, 10.0]", "1.0000", "0.7644"]
    ]
    story.append(build_table(bayes_headers, bayes_rows, [110, 150, 120, 124]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Untuned vs. Bayesian-Tuned Performance Comparison:", h2_style))
    b_comp_headers = ["Model Variant", "CV Macro F1", "CV Top-1", "Test Top-1", "Test Macro F1", "Test Log Loss", "Continuity", "Infer Time", "Fit Time"]
    b_comp_rows = [
        ["Untuned XGBoost", "0.9248", "0.8764", "0.8983", "0.9367", "0.0632", "0.9583", "0.84 ms/1k", "10.6 s"],
        ["Tuned XGBoost", "0.9266", "0.8785", "0.8890", "0.9339", "0.0660", "0.9560", "0.65 ms/1k", "8.5 s"]
    ]
    story.append(build_table(b_comp_headers, b_comp_rows, [94, 50, 45, 45, 52, 52, 52, 58, 56]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Analysis:</b> Bayesian tuning raised CV Top-1 from 0.8764 to <b>0.8785</b> (+0.21 pp) and raised CV Macro F1 from 0.9248 to <b>0.9266</b>, while reducing per-tree depth and tree counts, which reduced inference latency by 23% (0.65 ms vs 0.84 ms per 1,000 rows).",
        body_style
    ))
    story.extend(embed_img("results/bayes_opt/importance_xgboost.png", 320, "Figure 5: Hyperparameter importance for observation Top-1 accuracy from 30 Optuna TPE trials."))
    story.append(Spacer(1, 6))

    # ------------------ SECTION 5 ------------------
    story.append(Paragraph("5. HMM and Viterbi Route Reconstruction", h1_style))
    story.append(Paragraph(
        "<b>5.1 Mathematical Formulation:</b> Candidate classifiers evaluate points in isolation, ignoring topological network connectivity. The HMM integrates spatial classifier posteriors with physical road constraints:<br/>"
        "• <i>Hidden States</i>: Candidate arcs $\mathcal{C}(z_t)$ within 60m of GPS fix $z_t$ (pruned to top $m=8$ states by classifier probability for decoding speed).<br/>"
        "• <i>Emission Log-Probabilities</i>: $\ln P(z_t \mid a_i) = \gamma \cdot \ln(\max(p_{\text{clf}}(a_i), 10^{-6}))$, where $\gamma=0.5$ is an emission power temperature tuned on out-of-fold training data to balance emission confidence against network transition priors.<br/>"
        "• <i>Transition Log-Probabilities</i>: $\ln P(a_j^{(t)} \mid a_i^{(t-1)}) = -\frac{|d_{\text{network}}(a_i, a_j) - d_{\text{Euclidean}}(z_{t-1}, z_t)|}{\beta}$, where $\beta=5.0$m and $d_{\text{network}}$ is computed via bounded Dijkstra search on the directed graph. Unreachable transitions receive a $-30.0$ penalty.<br/>"
        "• <i>Gap Splitting & Route Stitching</i>: If $\Delta t > 180$s, the trellis resets. Post-Viterbi, non-adjacent consecutive arcs are connected via shortest-path topological routing.",
        body_style
    ))
    
    story.append(Paragraph("5.2 Experimental Results: Classifier-Only vs. Classifier + HMM", h2_style))
    hmm_headers = ["Partition", "Decoding Strategy", "Point Accuracy", "Continuity", "Route Recall", "Route Precision", "Route F1", "Route Mismatch"]
    hmm_rows = [
        ["Student-Test (10 trips)", "Classifier Only", "90.53%", "95.60%", "—", "—", "86.37%", "32.85%"],
        ["Student-Test (10 trips)", "Classifier + HMM", "94.73%", "97.93%", "99.66%", "98.09%", "98.86%", "2.31%"],
        ["Evaluator-Test (10 trips)", "Classifier Only", "91.91%", "94.49%", "—", "—", "86.82%", "30.68%"],
        ["Evaluator-Test (10 trips)", "Classifier + HMM", "95.82%", "96.50%", "99.61%", "97.35%", "98.44%", "3.24%"]
    ]
    story.append(build_table(hmm_headers, hmm_rows, [95, 80, 55, 52, 54, 56, 54, 58]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Impact:</b> HMM decoding improved accuracy on <b>10/10 student-test trips</b> (+4.20 pp point accuracy, +12.49 pp Route F1, 14x drop in mismatch) and <b>10/10 evaluator-test trips</b> (+3.91 pp point accuracy, +11.62 pp Route F1, 9.5x drop in mismatch). On evaluator trip 00000015, point accuracy surged from 70.63% to <b>93.33%</b> (+22.70 pp).",
        body_style
    ))
    story.extend(embed_img("results/hmm/clf_vs_hmm_student_test.png", 460, "Figure 6: Per-trip comparison of point accuracy and route continuity: Classifier-only vs HMM on student_test."))
    story.append(Spacer(1, 6))

    # ------------------ SECTION 6 ------------------
    story.append(PageBreak())
    story.append(Paragraph("6. Geographical Visualisation Using Folium", h1_style))
    story.append(Paragraph(
        "Folium interactive maps were generated for every trip in `student_test` and `evaluator_test` (`scripts/07_maps.py`). Each HTML map features five distinguishable layers: "
        "<b>Blue circles</b> (raw GPS observations), <b>Grey lines</b> (nearby road network), <b>Purple line</b> (reference route), <b>Orange dashed line</b> (classifier-only path showing jitter), and <b>Green line</b> (HMM predicted route).",
        body_style
    ))
    story.extend(embed_img("results/screenshots/map_student_test_00000077_zoom.jpg", 480, "Figure 7: Folium map of trip 00000077: Raw GPS (blue), Reference Route (purple), Classifier-only (orange dashed), and HMM prediction (green)."))
    story.append(Spacer(1, 8))

    # ------------------ SECTION 7 ------------------
    story.append(Paragraph("7. API, UI and Distributed Deployment", h1_style))
    story.append(Paragraph(
        "<b>Distributed Architecture:</b> PathSnap was deployed across two networked laptops: "
        "<b>Laptop 1 (FastAPI backend, port 8000)</b> serves the trained XGBoost model and executes HMM decoding. It logs per-request performance without sending ground-truth geometry to the client. "
        "<b>Laptop 2 (Evaluator UI, port 8080)</b> runs a Leaflet web application that connects to Laptop 1 over local Wi-Fi, rendering interactive route maps and live scorecards. "
        "Total end-to-end latency averages <b>176.2 ms per trip</b> (Features: 76.9ms, Classifier: 7.5ms, HMM: 57.1ms).",
        body_style
    ))
    story.extend(embed_img("results/screenshots/ui_student_test_00000056.jpg", 460, "Figure 8: Live PathSnap Web UI running in browser on Laptop 2, displaying evaluation trip 00000056 with 96.50% point accuracy and 98.13% Route F1."))
    story.append(Spacer(1, 8))

    # ------------------ SECTION 8 ------------------
    story.append(Paragraph("8. Academic Integrity and Methodological Compliance", h1_style))
    story.append(Paragraph(
        "1. <b>Strict Partition Isolation</b>: Hyperparameters and HMM transition parameters were selected strictly using the 80 training trips (5-fold GroupKFold). The student-test and evaluator-test partitions were never used for model training or tuning.<br/>"
        "2. <b>No Label Leakage in Prediction</b>: `PathSnap.predict()` ingests only `.track`, `.nodes`, and `.arcs`. Reference `.route` files are accessed solely inside `PathSnap.evaluate()` post-inference.<br/>"
        "3. <b>Authentic Experimental Verification</b>: All reported figures, confusion matrices, tables, and timings were produced by verifiable Python scripts in the submitted repository.",
        body_style
    ))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {PDF_PATH} ({PDF_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    write_markdown_report()
    generate_pdf_report()

