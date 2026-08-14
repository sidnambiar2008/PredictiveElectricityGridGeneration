# Deployment Handoff Notes

This document is for whoever sets up production infrastructure (ECS Fargate,
S3, Docker, Supabase/Postgres, Terraform, Flask/React) around this codebase.
It covers what the infrastructure needs to provide, not how to build it.

## 1. Critical: this pipeline assumes a persistent, shared filesystem

Every script here reads and writes plain files at repo-relative paths:

- `grid_data/raw/grid_history_{region}_v4.csv` — accumulated historical data
- `saved_models/lstm_grid_pulse_24h_{region}_v2.pt` and
  `saved_models/scaler_{region}_v2.pkl` — the trained model and the scaler it
  was trained with
- `evaluation_metrics/evaluation_metrics_{region}_v3.csv` — evaluation output
- `visuals/*.png` — generated plots

None of this is optional state — it's load-bearing. In particular,
`collect_history.py` works by **reading the existing CSV, appending the
new week's data, and writing it back**. If this ever runs in a fresh,
empty container with no access to the *same* files from the previous week
(which is the default behavior of an ECS Fargate task with no persistent
volume), `os.path.exists(save_path)` will be `False` every single run, and
the "incremental weekly collection" this script is built around silently
degrades into "3 years of accumulated history reset to 7 days, every
week" — the exact bug that was already fixed once in the code, just
reappearing at the infrastructure layer instead.

Same issue for `saved_models/`: `evaluate_model.py`, `run_back_test.py`,
and `production_forecast.py` all expect that week's trained model and
scaler to already exist on disk. If training and inference run in separate
ephemeral containers without a shared location, inference will fail to
find the model at all.

**What this means concretely:** whatever S3/EFS/volume strategy gets built
needs to keep `grid_data/`, `saved_models/`, and `evaluation_metrics/`
pointing at the same persistent location across every run, not just
within a single container's lifetime. This is the single most important
thing for the infrastructure to get right — everything else in this repo
already works locally as long as this holds.

## 2. Required environment variables

| Variable | Required by | Purpose |
|---|---|---|
| `EIA_API_KEY` | `src/ingestion/api_wrapper.py`, `src/ingestion/collect_history.py` | Authenticates all calls to the EIA-930 API. Missing it raises `ValueError` immediately, before any network call. |

That's the only one. Nothing else in the codebase reads from the environment.

This should be **the admin's own key**, not the one used in development —
EIA API keys are free but tied to whoever registers for one
(https://www.eia.gov/opendata/register.php), and it should be supplied
however the production environment manages secrets (e.g. an ECS task's
injected environment variables via Secrets Manager/Parameter Store), never
committed to the repo or baked into the Docker image.

## 3. Entry points

All scripts are run as modules from the repo root (this matters — they use
repo-relative paths, so the working directory must be the repo root):

| Script | Command | What it does | Reads | Writes                                                      |
|---|---|---|---|-------------------------------------------------------------|
| `src/ingestion/collect_history.py` | `python -m src.ingestion.collect_history` | Fetches the last 7 days of data for all 7 regions, appends/dedupes into each region's history CSV | EIA API | `grid_data/raw/*.csv`                                       |
| `src/training/train.py` | `python -m src.training.train` | Trains (retrains) the LSTM for all 7 regions from scratch | `grid_data/raw/*.csv` | `saved_models/*.pt`, `saved_models/*.pkl`                   |
| `src/evaluation/evaluate_model.py` | `python -m src.evaluation.evaluate_model` | Scores each region's trained model against its held-out validation split | `grid_data/raw/*.csv`, `saved_models/*` | `evaluation_metrics/*.csv`                                  |
| `src/evaluation/run_back_test.py` | `python -m src.evaluation.run_back_test` | Ad hoc: backtest one region/fuel/date combination and plot it | EIA API, `saved_models/*` | `visuals/*.png`                                             |
| `src/inference/production_forecast.py` | `python -m src.inference.production_forecast` | Produces the actual 24-hour forecast + CO2/cleanliness stats for one region | EIA API, `saved_models/*` | `visuals/*.png`; logs to stderr (via `logging`, not stdout) |

Regions are the 7 EIA balancing authorities used throughout: `CISO, PJM,
SWPP, ERCO, MISO, ISNE, NYIS`.

## 4. Intended schedule

This was an explicit production decision made mid-project: **acquire a new
week of data and retrain the model every week.**

- `collect_history.py` → weekly, and must run *before* `train.py` each week
  (training reads whatever is currently in `grid_data/raw/`)
- `train.py` → weekly, immediately after `collect_history.py`
- `evaluate_model.py` → not strictly time-critical, but makes sense to run
  right after each retrain to get fresh metrics
- `run_back_test.py` → manual/on-demand only — this is an analysis tool, not
  something that should be scheduled
- `production_forecast.py` → currently a script that forecasts one
  hardcoded region (`PJM`) per invocation via its '__main__'. To
  actually serve forecasts on demand (the Flask/React part of the roadmap),
- call `predict_24_hours_hours_ahead(region_id)` directly per request instead
  of relying on the script. It returns a dict (`region_id`, `current`
  cleanliness/CO2 snapshot, `summary` day-ahead aggregates, `forecast_df`,
  `baseline_df`) in addition to its existing logging/plotting side effects —
  a Flask route can serve that dict as JSON directly.

## 5. Known limitations

- **Solar and Hydro forecasting is weak.** Documented with a real case study
  in the README (Case Study 2) — the model underperforms the diurnal-mean
  baseline on these, sometimes badly. Not something introduced by a bug;
  it's an honest limitation of the current model.
- **Training uses a fixed 15 epochs, no early stopping.** There's no
  validation-loss-based cutoff — every region trains for exactly the same
  number of epochs regardless of whether it's still improving.
- **Training and evaluation run sequentially, one region at a time,** not in
  parallel. All 7 regions are looped over in a single process. This affects
  how long the weekly retrain job actually takes end to end, worth knowing
  when sizing the ECS task's timeout/resources.
- **CO2/carbon-intensity estimates in `production_forecast.py` have a known
  edge case:** for CISO specifically, the "Other" fuel bucket absorbs
  Geothermal generation (see the merge logic in `api_wrapper.py`), but the
  CO2 calculation still prices "Other" as fossil-equivalent (920 lbs/MWh).
  This somewhat overstates CISO's true carbon intensity. Documented inline
  in the code, not fixed.
- **API retry logic is fixed, not adaptive:** 3 attempts, 5-second delay
  between each, no exponential backoff.
- **The Winter Storm Elliott case study (README Case Study 4) predates the
  model's training window** — it's included as an out-of-distribution
  stress test, not a claim of real predictive skill on that event.

## 6. Local setup / testing

```
pip install .          # runtime dependencies only
pip install .[dev]     # adds pytest, for running the test suite
pytest                 # runs tests/ — all fully mocked, no EIA_API_KEY needed
```

A GitHub Actions workflow (`.github/workflows/tests.yml`) runs the test
suite automatically on every push/PR to `main`.
