# PathSnap Deliverables Checklist
## PES University Machine Learning Technical Report & Submission Guidelines Audit

Every mandatory requirement from the submission document is mapped below to its verified file in this repository:

### Deliverable 1: GitHub Repository (Public)
Repository: **https://github.com/ack-chai/pathsnap**

| Required Component | What must be included | Repository Location | Verified Status |
|---|---|---|:---:|
| **Data processing** | Source code used to read, inspect, clean and preprocess the raw dataset | `pathsnap/io.py`<br/>`pathsnap/geo.py`<br/>`clean_data.py`<br/>`scripts/01_eda.py` | Complete |
| **Feature engineering** | Complete source code for feature extraction (23 features) | `pathsnap/features.py`<br/>`scripts/02_build_dataset.py` | Complete |
| **Model training** | Training scripts/notebooks for every model experimented with (10 models) | `pathsnap/models.py`<br/>`scripts/03_compare_models.py`<br/>`scripts/05_export_model.py` | Complete |
| **Model evaluation** | Code for comparison, evaluation metrics, plots, confusion matrices | `pathsnap/metrics.py`<br/>`scripts/03_compare_models.py`<br/>`scripts/08_fit_diagnostics.py` | Complete |
| **Hyperparameter optimisation** | Bayesian optimisation code, search configurations, experiment results | `scripts/04_bayes_opt.py`<br/>`results/bayes_opt/` | Complete |
| **HMM and Viterbi** | Source code for route reconstruction and decoding | `pathsnap/hmm.py`<br/>`pathsnap/pipeline.py`<br/>`scripts/06_tune_hmm.py` | Complete |
| **API** | Complete backend source code, endpoints, model-loading logic | `api/main.py` | Complete |
| **UI** | Complete evaluator interface source code | `ui/index.html`<br/>`ui/app.js`<br/>`ui/style.css` | Complete |
| **Deployment** | Configuration files, dependencies, running instructions | `HOW_TO_RUN.md`<br/>`README.md`<br/>`requirements.txt` | Complete |
| **Visualisations** | Folium map-generation code and other plotting scripts | `scripts/07_maps.py`<br/>`scripts/09_concept_figures.py`<br/>`deliverables/plots/` | Complete |
| **Supporting results** | Relevant CSV files, experiment logs, configurations, summaries | `results/`<br/>`deliverables/tables/` | Complete |

---

### Deliverable 2: Technical Report (PDF)
- **Primary PDF**: `deliverables/PathSnap_Technical_Report.pdf` (12 pages, fully formatted)
- **Source Markdown**: `deliverables/PathSnap_Technical_Report.md`
- **Demo & Defense Guide**: `deliverables/PathSnap_Demo_Guide.pdf`

| Section | Guideline Requirement | Coverage in Report |
|---|---|---|
| **2.2 EDA & Preprocessing** | Missing-value analysis, preprocessing steps, trajectory plots, road network connectivity, Folium map | Pages 2–3, Section 2 (Tables, Fig 1 & Fig 2) |
| **3.1 Model Comparison** | Consolidated table with Model, Family, Key Config, CV Macro F1, Top-1, Continuity, Test F1, Log Loss, Inference Time | Page 4, Section 3.1 (Table covering 10 models) |
| **3.2 Compulsory Questions** | Detailed subsections for EVERY model covering Understanding (Q1, Q2), Training (Q5, Q6), Results & Overfitting (Q7, Q8, Q9) | Pages 5–7, Section 3.2 (All 10 models addressed individually) |
| **3.3 Visualisations** | CV F1, Student Test Top-1/F1, Log Loss, Confusion matrix grid, MLP epoch loss history, Continuity | Pages 7–8, Section 3.3 (Figures 3, 4, 5, 6) |
| **4. Bayesian Optimisation** | Hyperparameters, distributions, objective, 30 trials, best values, tuned vs untuned comparison table | Page 8, Section 4 (Table & Figures 7, 8) |
| **5. HMM & Viterbi** | Hidden states, emissions, transitions, gap splitting, route stitching, experimental results comparison table | Pages 9–10, Section 5 (Table & Figure 9) |
| **6. Folium Visualisation** | 5 distinguishable layers, legend, zoom level, screenshot in PDF, interactive HTML files committed | Page 10, Section 6 (Figure 10 & interactive files) |
| **7. API, UI & Deployment** | Screenshots of UI and successful API responses on CMD, distributed setup description | Page 11, Section 7 (Figures 11 & 12) |
| **8. Academic Integrity** | Declarations on data partition isolation, no label leakage in prediction, authentic experiment outputs | Page 12, Section 8 |
