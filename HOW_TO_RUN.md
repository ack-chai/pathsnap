# PathSnap: how to run it

PathSnap takes a noisy GPS trip and a road network and works out which roads the
vehicle actually drove on. It does this with an XGBoost candidate classifier
followed by an HMM/Viterbi decoder. `README.md` has the full method and results.
This page covers getting it running.

Everything you need is in this folder: the code, the trained model (`artifacts/`),
the dataset (`data/`: 80 training trips and 10 student-test trips), and all the
results, plots and maps (`results/`).

---

## 0. Just look at the results (no setup)

Double-click any of these to open them in a browser:

| What | File |
|---|---|
| Predicted route vs reference, per trip (interactive maps) | `results/maps/student_test_*.html` |
| EDA map (GPS + road network + reference route) | `results/eda/map_trip_00000001.html` |
| Model comparison table | `results/models/comparison.csv` |
| Classifier vs HMM, per trip | `results/hmm/classifier_vs_hmm_per_trip.csv` |
| Plots | `results/*/plots/*.png`, `results/hmm/*.png`, `results/concepts/*.png` |

In a map, use the layer box at the top right to switch layers:
- **Purple** is the true route.
- **Green** is our HMM prediction.
- **Orange dashed** is the classifier alone. It is hidden at first; tick its box to show it.

The map background needs internet.

---

## 1. One-time setup

You need **Python 3.11 or newer**. Check with `python3 --version` on Mac or
`py --version` on Windows.

Open a terminal **inside this folder**, then run the commands for your system.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install libomp        # macOS only; XGBoost/LightGBM need it
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activate script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

After this, run `source .venv/bin/activate` (Mac) or `.venv\Scripts\Activate.ps1`
(Windows) each time you open a new terminal.

---

## 2. Run the live demo (API + web UI)

You need **two terminals**, both opened in this folder with the venv activated.

**Terminal 1: the backend API.** It loads the model and predicts routes.

macOS / Linux:
```bash
PATHSNAP_EVAL_PARTITION=student_test uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Windows (PowerShell):
```powershell
$env:PATHSNAP_EVAL_PARTITION="student_test"; uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Wait until it prints `Uvicorn running on http://0.0.0.0:8000`.

**Terminal 2: the web UI.**

```bash
python3 -m http.server 8080 --directory ui
```

On Windows, use `py` in place of `python3`.

Now open **http://localhost:8080** in a browser and click a trip in the list. The
map shows the GPS points and the predicted route, and the panel shows the accuracy.
Terminal 1 prints one log line per request.

`PATHSNAP_EVAL_PARTITION=student_test` makes the API serve the 10 student-test
trips. Without it, the API looks for `data/evaluator_test/`, which isn't
included.

**Two-laptop setup:**
- Run Terminal 1 on laptop A. Get its IP with `ipconfig getifaddr en0` (Mac) or
  `ipconfig` (Windows).
- On laptop B, open `http://localhost:8080/?api=http://<LAPTOP_A_IP>:8000`.
- Both laptops must be on the same Wi-Fi, and port 8000 must be allowed through
  laptop A's firewall.

---

## 3. Regenerate the maps

```bash
python scripts/07_maps.py student_test             # all 10 student-test trips
python scripts/07_maps.py student_test 00000077    # just one trip
```

The output goes to `results/maps/`. Open the `.html` file to see it.

---

## 4. Reproduce every experiment from scratch (optional)

Run these from this folder, in order. Each step writes to `results/`, and some also
write to `cache/` or `artifacts/`.

```bash
python scripts/01_eda.py              # dataset stats, plots, EDA map          (~1 min)
python scripts/02_build_dataset.py    # candidate + feature tables -> cache/   (a few min)
python scripts/03_compare_models.py   # 10 models, 5-fold group CV + test      (slowest, ~15-30 min)
python scripts/04_bayes_opt.py xgboost --trials 30   # Optuna tuning          (~10-20 min)
python scripts/05_export_model.py     # writes artifacts/model.joblib
python scripts/06_tune_hmm.py         # HMM decoder tuning, classifier vs HMM
python scripts/07_maps.py             # Folium maps
python scripts/08_fit_diagnostics.py  # overfitting check (train vs CV vs test)
python scripts/09_concept_figures.py  # explanatory figures
```

Times are rough and depend on the laptop. Scripts 03 and later need step 02 to
have been run first, because they read `cache/`. Random seeds are fixed (42), so
the numbers should match `README.md`.

> Steps 04-06 **overwrite** `artifacts/`. If you only want the demo, skip section 4.
> The trained model is already included.

---

## Folder map

```
api/            FastAPI backend (POST /predict, GET /trips, GET /health)
ui/             browser UI (Leaflet map)
pathsnap/       core library: geometry, road graph, features, models, HMM
scripts/        01-09, the experiment pipeline (see section 4)
artifacts/      trained XGBoost model + decoder config used by the API
data/           train/ and student_test/ trips (.track .nodes .arcs .route)
results/        every CSV, plot and map from our runs
README.md       method, metrics and results in detail
```

## Common problems

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: pathsnap` | Run commands from this folder, not from inside `scripts/`. |
| `Library not loaded: libomp.dylib` (Mac) | `brew install libomp` |
| UI connects but the trip list is empty | You forgot `PATHSNAP_EVAL_PARTITION=student_test` when starting the API (section 2). |
| UI says it can't connect | Check that Terminal 1 is still running and the API box says `http://localhost:8000`, then press Connect. |
| Map is grey with no streets | No internet. Tiles come from OpenStreetMap; the routes still draw. |
| `pip install` fails on a version | Use Python 3.11-3.13. Very new or old versions may not have wheels yet. |
