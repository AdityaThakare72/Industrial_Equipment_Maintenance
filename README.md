env with conda => industrial_mantainance


# Industrial_Equipment_Maintenance

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

An end-to-end MLOps pipeline for predictive maintenance, automating fault detection and model lifecycle management.

**Live demo:** https://industrial-equipment-maintenance.streamlit.app/ (Streamlit dashboard) →
FastAPI inference API on Render. Both run on free tiers and sleep when idle, so the
first scan can take up to a minute.

## Results

XGBoost classifier on 7,672 sensor readings (10% faulty), evaluated on a stratified
20% hold-out set (1,535 rows, 153 faults). Numbers from `reports/metrics.json`:

| Metric | Value |
|---|---|
| Accuracy | 98.4% |
| Fault precision | 93.8% |
| Fault recall | 89.5% |
| Fault F1 | 91.6% |

What the model learned matches the data: healthy machines in the dataset stay
within roughly 38–109 °C, ≤ 62 PSI and ≤ 3.7 mm/s vibration, and the model flags
readings outside those bands as faulty.

### Modelling notes

- **No leakage:** the scaler/encoder and the class-imbalance ratio are fitted on the
  training split only; `train.py` and `predict.py` recreate the same split from
  `params.yaml`.
- **Class imbalance:** handled with XGBoost `scale_pos_weight` (≈ 9). Across 5×5
  repeated stratified CV it trades ~4 points of precision for ~1.3 points of recall
  (F1 0.898 vs 0.910 without weighting), a reasonable choice when a missed fault
  costs more than a false alarm. Resampling (e.g. SMOTE) isn't needed.
- **Remaining misses:** the faults the model misses have sensor readings that look
  entirely normal (e.g. 87 °C, 34 PSI, 0.9 mm/s), so they can't be separated with
  these four sensors. More signals (trends over time, maintenance history) would be
  needed, not a different classifier.
- **Out-of-range inputs:** the API returns `warnings` when a reading is outside the
  training range or a category wasn't seen in training, instead of silently
  extrapolating.

## Running locally

```bash
dvc repro                                   # ingest → featurize → train → evaluate
uvicorn app.main:app --port 8000            # inference API
API_URL=http://127.0.0.1:8000 streamlit run app/dashboard.py
```

## Project Organization 
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│our tool set
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         industrial_maintenance and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── industrial_maintenance   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes industrial_maintenance a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

