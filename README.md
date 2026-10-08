# Freight rate prediction

Predict total posted shipment rates for all 12,000 final loads and the fixed December scenario.

The submitted model is core CatBoost with RMSE loss, 425 trees, and no market index or quote signal. September-October comparison results influenced this choice; they are selection evidence, not an independent final test. Actual November-December accuracy is unavailable locally.

## Contents

- `solution.py`: reusable final preprocessing, training, evaluation, and inference.
- `notebooks/freight_rate_prediction_clean.ipynb`: concise primary notebook; reruns data checks, baselines, saved-model predictions, and the official scorer, and presents clearly labeled historical experiments.
- `notebooks/freight_rate_prediction.ipynb`: original exploration and full experiment history, preserved unchanged.
- `data/`: original four CSVs, unchanged.
- `artifacts/final_model.cbm`: final trained model; inference needs no GPU.
- `model_config.json`: recorded training parameters and final iteration count.
- `outputs/`: recorded metrics, out-of-fold predictions, diagnostics, and December predictions.
- `validation_predictions.csv`: required 12,000-row submission, exactly `load_id,predicted_rate`.
- `score.py`: original organizer scorer, unchanged.
- `scorer_results/candidate_december.png`: required generated chart.
- `freight_rate_report.pdf`: approach, actual results, limitations, and chart.
- `build_submission.py`: report generation and packaging utilities.

## Setup

Use the Python version recorded in `environment.json`. In Colab install the requirements directly; locally an isolated environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Verify and reproduce submitted predictions

```bash
python solution.py --mode verify
python solution.py --mode predict
python score.py --predictions validation_predictions.csv --december-predictions outputs/december_predictions.csv --output-dir scorer_results
python build_submission.py --project . --report-only
```

`verify` reloads the saved model and checks every prediction against the existing CSVs without overwriting predictions. `predict` regenerates both CSVs. The scorer checks formatting and the fixed scenario, not prediction accuracy.

## Retrain or repeat the later-period evaluation

```bash
python solution.py --mode train --task-type GPU
python solution.py --mode evaluate --task-type GPU
```

Training refits all 48,000 labeled loads using the recorded 425-tree budget, replaces the saved final model, and regenerates predictions. Add `--select-iterations` to repeat the final inner chronological stopping step. Evaluation trains through August, uses its last 14 days for inner early stopping, refits through August, and writes `outputs/reproduced_reserved_metrics.csv` for September-October. It does not overwrite the archived original comparison metrics.

CPU training is supported with `--task-type CPU`, but training hardware and GPU nondeterminism can produce different results. Using the archived model reproduces submitted predictions. Record your own runtime; no universal runtime guarantee is made.

## Method and limitations

Chronological development folds: Jan-Feb -> Mar-Apr; Jan-Apr -> May-Jun; Jan-Jun -> Jul-Aug. The later comparison uses Jan-Aug -> Sep-Oct. Preprocessing preserves all rows and labels, replaces nonpositive weights with missing values, and adds missing/invalid flags. CatBoost handles numerical missing values and categorical inputs.

Core features are pickup, delivery, equipment, coordinates, distance, weight, weight-quality flags, and weekday. Final predictions are floored at one dollar, consistently with evaluation. December coordinates come from the development city mapping. Because weekday is the only calendar feature, the fixed chart repeats weekly and does not represent a learned December trend or holiday surcharge.

The market/MAE candidate won earlier development comparisons; the core/RMSE candidate was chosen after its stronger September-October results. An unfamiliar-city stress test was performed on the market/MAE candidate, not the final core model. Do not attribute that candidate's stress scores to the final model. See the report for complete results and limitations.

## Concise notebook

Open `notebooks/freight_rate_prediction_clean.ipynb` and run all cells in order. It resolves the project folder automatically when opened within this repository or when the complete package is extracted to `/content/freight_submission` in Colab. For another location, set `PROJECT_ROOT` in the first code cell. Set `INSTALL_DEPENDENCIES = True` for initial Colab setup; restart the runtime if requested, then run from the beginning.

The default run needs no GPU. It recomputes data summaries and chronological baselines, calculates historical experiment metrics from recorded predictions, loads the submitted model, and verifies all regenerated predictions against the archived CSVs. New results go into `notebook_results/`; the original model and submission files are preserved. The original notebook contains full training code for the alternative models. The concise notebook is not a fresh rerun of every experiment.

Optional core-model retraining is controlled by `RUN_RETRAINING` and `TRAIN_TASK_TYPE`. Its models and metrics are saved separately under `notebook_results/retraining/`. The prediction/export cells continue using the archived submitted model. No optional training is needed to reproduce the submission.

## Full experimental history

Open `notebooks/freight_rate_prediction.ipynb` in Colab to inspect the complete exploration, baseline comparisons, CatBoost experiments, and diagnostics. This original history contains runtime switching, checkpoint restoration, and unsuccessful experiments with their corrections; it is not the streamlined fresh-runtime entry point. Use the concise notebook or standalone `solution.py` for the final execution path.

## Submission

Submit `validation_predictions.csv` and `freight_rate_report.pdf`, publish this repository, and add your own Loom link here:

Loom: ADD_YOUR_LINK
