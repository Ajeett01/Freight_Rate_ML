"""Build and verify the submission inside the user's existing Colab project."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak


def number(value, decimals=2):
    if pd.isna(value):
        return "-"
    return f"{float(value):,.{decimals}f}"


def development_metrics(outputs):
    filenames = [
        "baseline_oof_predictions.csv", "linear_oof_predictions.csv",
        "catboost_core_oof_predictions.csv", "catboost_market_oof_predictions.csv",
        "catboost_market_mae_oof_predictions.csv", "catboost_market_mae_scaled_oof_predictions.csv",
        "quote_oof_predictions.csv",
    ]
    records = []
    for name in filenames:
        path = outputs / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing recorded experiment: {path}")
        frame = pd.read_csv(path)
        for model, group in frame.groupby("model"):
            if not group["load_id"].is_unique or len(group) != 29222:
                raise ValueError(f"Unexpected OOF rows for {model}")
            actual = group["actual_rate"].to_numpy()
            predicted = group["predicted_rate"].to_numpy()
            if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
                raise ValueError(f"Nonfinite OOF values for {model}")
            error = predicted - actual
            records.append({"model": model, "MAE": np.abs(error).mean(),
                            "RMSE": np.sqrt(np.mean(error ** 2)),
                            "WAPE_pct": 100 * np.abs(error).sum() / np.abs(actual).sum(),
                            "bias": error.mean()})
    return pd.DataFrame(records).sort_values("MAE")


def make_report(repo, metadata, config, dev_metrics):
    outputs = repo / "outputs"
    reserved = pd.read_csv(outputs / "reserved_metrics.csv")
    final_row = reserved.set_index("model").loc["chart_core_rmse"]
    base_row = reserved.set_index("model").loc["equipment_mileage"]
    city_stress = pd.read_csv(outputs / "unfamiliar_city_stress_metrics.csv")
    from solution import load_inputs
    inputs = load_inputs(repo)
    train, inference = inputs["train"], inputs["inference"]

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallText", fontName="Helvetica", fontSize=8, leading=11, spaceAfter=5))
    styles.add(ParagraphStyle(name="TableText", fontName="Helvetica", fontSize=7.4, leading=9))
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 13
    styles["BodyText"].spaceAfter = 8
    styles["Title"].textColor = colors.HexColor("#064A56")
    story = []

    def p(text, style="BodyText"):
        story.append(Paragraph(escape(str(text)), styles[style]))

    def heading(text):
        p(text, "Heading2")

    def table(headers, rows, widths):
        data = [[Paragraph(escape(str(x)), styles["TableText"]) for x in row] for row in [headers] + rows]
        item = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCECEF")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#064A56")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7F8")]),
        ]))
        story.extend([item, Spacer(1, 10)])

    p("Freight Rate Prediction", "Title")
    p("Approach, validation, final model, and required December chart", "Heading2")
    p("The task is supervised regression: predict each shipment's total posted rate in dollars. The final solution produces one positive rate for every required load ID and a controlled December scenario using the same trained estimator.")
    heading("Final decision and evidence")
    p(f"The submitted model is core CatBoost with RMSE loss and {metadata['trees']} trees, fitted on all {len(train):,} labeled loads. It uses shipment and geographic features plus weekday; it excludes market_index and quote_signal. Its January-August-trained counterpart achieved MAE ${final_row['MAE']:,.2f}, RMSE ${final_row['RMSE']:,.2f}, WAPE {final_row['WAPE_pct']:.2f}%, and signed bias ${final_row['bias']:,.2f} on September-October.")
    reduction = 100 * (1 - final_row["MAE"] / base_row["MAE"])
    p(f"The equipment-mileage baseline had MAE ${base_row['MAE']:,.2f} on the same period: the core model reduced MAE by {reduction:.1f}%. These are historical selection results, not known accuracy on the 12,000 final loads.")
    p("Disclosure: September-October was initially reserved for a later comparison but was already profiled during EDA. Its model results then changed the final selection from the market/MAE candidate to the core/RMSE candidate. It therefore contributed to model selection and must not be described as an independent final test. November-December outcomes remain hidden.")
    heading("Data and quality checks")
    table(["Dataset", "Rows", "Coverage / role"], [
        ["Labeled development", f"{len(train):,}", "January-October 2025; posted_rate available"],
        ["Final inference", f"{len(inference):,}", "November-December 2025; target unavailable"],
        ["Prediction template", "12,000", "Required load IDs and blank predicted_rate"],
        ["December scenarios", "31", "One fixed shipment; date changes daily"],
    ], [135, 60, 315])
    quality_rows = []
    for label, frame in [("Development", train), ("Final inference", inference)]:
        weights = pd.to_numeric(frame["weight"], errors="raise")
        quality_rows.append([label, int(weights.isna().sum()), int(weights.lt(0).sum()), int(frame["market_index"].isna().sum())])
    table(["Dataset", "Missing weights", "Negative weights", "Missing market index"], quality_rows, [130, 110, 120, 150])
    p("Negative/nonpositive weights are replaced with missing values. Separate flags preserve whether weight was originally missing or invalid. CatBoost handles numerical missing values. No development or final rows were removed; no evaluation targets were clipped. The audit found no duplicate IDs or feature rows in the main files. Supplied coordinates were preserved rather than replaced with external geocoding.")

    story.append(PageBreak())
    heading("Chronological validation and leakage prevention")
    p("The final task predicts two future months, so evaluation uses expanding training windows with two-month future periods. Calendar cutoffs keep all loads from a date on the same side. Random splitting would mix temporal regimes, especially the unstable quote signal.")
    table(["Role", "Training", "Evaluation", "Train rows", "Eval rows"], [
        ["Fold A", "Jan-Feb", "Mar-Apr", "9,255", "9,855"],
        ["Fold B", "Jan-Apr", "May-Jun", "19,110", "9,696"],
        ["Fold C", "Jan-Jun", "Jul-Aug", "28,806", "9,671"],
        ["Later comparison", "Jan-Aug", "Sep-Oct", "38,477", "9,523"],
        ["Final refit", "Jan-Oct", "Nov-Dec (unlabeled)", "48,000", "12,000"],
    ], [105, 85, 150, 85, 85])
    p("For each CatBoost experiment, the final 14 calendar days inside the training window select the number of trees with early stopping (2,000-tree maximum; patience 100). The model is then refitted on the entire outer training partition. Outer evaluation labels are not used for early stopping. Scaling and fitted baseline statistics are learned inside each fitting partition. IDs and target-derived rate-per-mile diagnostics are excluded from model features.")
    p("MAE is the provisional primary selection metric; RMSE, WAPE, and signed bias reveal other behavior. The organizer's hidden ranking metric was not specified. WAPE is total absolute error divided by total actual rates; it is not an accuracy percentage. Pooled scores are recomputed from per-load out-of-fold predictions, not averaged RMSEs.")
    heading("Recorded development experiments: March-August")
    labels = {
        "constant_median": "Constant median", "equipment_mileage": "Equipment x mileage",
        "ridge": "Ridge", "huber": "Linear Huber", "catboost_core": "Core CatBoost / RMSE [final recipe]",
        "catboost_market": "Market CatBoost / RMSE", "catboost_market_mae": "Market / MAE, unscaled [underfit]",
        "catboost_market_mae_scaled": "Market / MAE, scaled", "catboost_market_mae_quote": "Market / MAE, scaled + quote",
    }
    rows = [[labels.get(r["model"], r["model"]), number(r["MAE"]), number(r["RMSE"]), number(r["WAPE_pct"]), number(r["bias"])] for _, r in dev_metrics.iterrows()]
    table(["Model", "MAE ($)", "RMSE ($)", "WAPE (%)", "Bias ($)"], rows, [210, 75, 75, 75, 75])
    p("These experiments share 29,222 evaluated loads. Adding market index improved early-period results. A GPU Ordered/Gradient MAE run on raw dollar targets severely underfit; scaling the target by 1,000 and reversing that scaling at inference corrected its optimization behavior. This target scaling belongs to the experimental MAE models; the final RMSE model predicts dollars directly.")
    p("Adding quote_signal worsened pooled MAE and two of three folds. Its relationship with rate per mile changes sign by month and is near zero in August. It was excluded for observed instability. The data does not establish its provenance, so the report does not assert proven target leakage.")

    story.append(PageBreak())
    heading("Later-period selection and generalization")
    rows = [[r["model"], number(r["MAE"]), number(r["RMSE"]), number(r["WAPE_pct"]), number(r["bias"])] for _, r in reserved.iterrows()]
    table(["January-August-trained model", "MAE ($)", "RMSE ($)", "WAPE (%)", "Bias ($)"], rows, [210, 75, 75, 75, 75])
    p("The market/MAE candidate led on earlier folds but degraded in September-October. The core RMSE model was selected for stronger performance in this most recent period and for a single consistent prediction path across both required input formats. This is a recency-based engineering decision; it does not establish that core RMSE is universally best. No further parameter search was performed after this decision.")
    heading("Unfamiliar-city stress test: scope and limitation")
    p("There are 1,447 final loads involving a development-unseen city (12.06%) and 1,461 loads on unseen directional lanes. A supplementary fold-C stress test withheld four cities and all touching loads from its fitting data, then evaluated later loads involving those cities. The cities were chosen with a fixed seed without consulting their rates.")
    selected = city_stress.loc[city_stress["city_group"].eq("withheld_city")]
    table(["Stress-test model", "Loads", "MAE ($)", "RMSE ($)"], [
        [r["model"], int(r["loads"]), number(r["MAE"]), number(r["RMSE"])] for _, r in selected.iterrows()
    ], [290, 60, 80, 80])
    p("Important scope distinction: this stress test evaluated the market/MAE candidate, not the final core RMSE model. Its exact scores cannot be claimed as the final model's unseen-city performance. Removing cities also reduces training size, so the comparison does not isolate category novelty alone. The final core model includes geographic features and supports unseen categories, but its unseen-city accuracy remains a limitation requiring further validation in a production setting.")
    heading("Alternative methods and stopping decision")
    p("Constant pricing, equipment-mileage, Ridge, linear Huber, multiple CatBoost feature sets, and RMSE versus MAE objectives were evaluated. LightGBM, XGBoost, random forests, neural networks, and aggregate time-series methods were considered but not trained. They are alternatives, not failed experiments. Model search was bounded to avoid adding complexity without evidence. An additional multi-seed study was not performed; GPU training is not bitwise deterministic.")

    story.append(PageBreak())
    heading("Final model and fixed December chart")
    p(f"The final estimator uses {metadata['trees']} trees, depth {config['parameters']['depth']}, learning rate {config['parameters']['learning_rate']}, RMSE loss, and seed {config['parameters']['random_seed']}. The last 14 labeled days (October 18-31) selected its tree budget before refitting on all January-October rows. The final tree count differs from the earlier-period model because its inner training window differs.")
    p("Features: pickup and delivery city, equipment, four supplied coordinates, distance, cleaned weight, original-missing and invalid-weight flags, and weekday. No market index, quote signal, raw ID, target-derived feature, full-date trend, or holiday feature is used. No target scaling is applied to the final RMSE model.")
    p("The December file lacks coordinates. They are reconstructed from the consistent development-only city mapping; its supplied 360-mile distance remains unchanged. The same saved final model predicts both the 12,000 final loads and these scenarios.")
    chart = repo / "scorer_results" / "candidate_december.png"
    width, height = ImageReader(str(chart)).getSize()
    story.append(Image(str(chart), width=510, height=510 * height / width))
    story.append(Spacer(1, 10))
    p("This is the chart generated by the original score.py. Fixed inputs are Lexington to Fort Wayne, 360 miles, Dry Van, and 32,000 lb, for December 1-31, 2025. Only weekday changes in the model's feature representation. Identical weekdays therefore repeat the same prediction, with at most seven distinct input combinations.")
    p("The weekly repetition is expected model behavior, not a plotting defect. The model cannot represent a December-wide trend or a Christmas-specific surcharge without corresponding features and evidence. The chart is a controlled scenario, not a validated daily market forecast. There are no labeled December observations.")

    story.append(PageBreak())
    heading("Submission verification and practical limitations")
    p("The packaged standalone script reloads the final model and reconstructs predictions for every final load and December scenario. Its predictions are checked against the submitted CSVs to a $0.000001 tolerance, allowing six-decimal CSV rounding. The official scorer then validates exactly 12,000 required unique IDs, column order, finite positive rates, and the 31 fixed December rows, and regenerates the required chart.")
    p("Scorer success establishes file/scenario validity, not predictive accuracy. November-December labels remain unavailable. The final model's accuracy on those 12,000 loads is unknown until the organizer evaluates it.")
    p("Large individual errors remain: RMSE is much greater than MAE. The earlier equipment-mileage diagnostic found roughly 1% of evaluated loads accounted for about 88% of squared error; that concentration statistic belongs to that baseline and was not measured for the final model. High-rate labels were retained. Tail errors, potential temporal drift, semantic accuracy of supplied coordinates, and weekday-only seasonality limit business deployment claims.")
    heading("Reproduction")
    p("The archive includes the executed notebook, solution.py, pinned runtime dependencies, original input CSVs, unchanged score.py, recorded experiments, final model, model configuration, predictions, and this report. The notebook records the full exploration and model-development history. The standalone script supports final inference, saved-model verification, refitting, and reconstruction of the core model's later-period evaluation.")
    for command in [
        "python -m pip install -r requirements.txt",
        "python solution.py --mode verify",
        "python solution.py --mode predict",
        "python score.py --predictions validation_predictions.csv --december-predictions outputs/december_predictions.csv --output-dir scorer_results",
        "python solution.py --mode train --task-type GPU",
        "python solution.py --mode evaluate --task-type GPU",
    ]:
        p(command, "SmallText")
    p("Prediction from the saved model does not require a GPU. Refitting uses the recorded final tree count by default; --select-iterations repeats the inner chronological stopping procedure. GPU training can vary slightly between runs, and changing training hardware can change defaults and results. Archived metrics describe the original recorded runs.")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#5B6B70"))
        canvas.drawString(42, 24, "Freight rate prediction | Historical selection results; final labels unavailable")
        canvas.drawRightString(A4[0] - 42, 24, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(repo / "freight_rate_report.pdf"), pagesize=A4,
                            rightMargin=42, leftMargin=42, topMargin=36, bottomMargin=42,
                            title="Freight Rate Prediction - Approach and Validation")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


README = """# Freight rate prediction

Predict total posted shipment rates for all 12,000 final loads and the fixed December scenario.

The submitted model is core CatBoost with RMSE loss, 425 trees, and no market index or quote signal. September-October comparison results influenced this choice; they are selection evidence, not an independent final test. Actual November-December accuracy is unavailable locally.

## Contents

- `solution.py`: reusable final preprocessing, training, evaluation, and inference.
- `notebooks/freight_rate_prediction.ipynb`: executed exploration and experiment history.
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

## Full experimental history

Open the included notebook in Colab to inspect or rerun exploration, baseline comparisons, CatBoost experiments, and diagnostics. GPU switching required a checkpoint and helper-function restoration. The notebook retains unsuccessful experiments and their corrections. The standalone `solution.py` is the streamlined final execution path and does not require running experimental cells.

## Submission

Submit `validation_predictions.csv` and `freight_rate_report.pdf`, publish this repository, and add your own Loom link here:

Loom: ADD_YOUR_LINK
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/content/freight_rate_project"))
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--report-only", action="store_true", help="Rebuild the PDF inside an already verified submission directory")
    parser.add_argument("--destination", type=Path, default=Path("/content/freight_submission"))
    args = parser.parse_args()
    project, repo = args.project.resolve(), args.destination.resolve()
    outputs = project / "outputs"
    if args.report_only:
        metadata = json.loads((project / "artifacts" / "final_model_metadata.json").read_text())
        config = json.loads((project / "model_config.json").read_text())
        make_report(project, metadata, config, development_metrics(outputs))
        print(f"Rebuilt {project / 'freight_rate_report.pdf'}")
        return
    if args.notebook is None or not args.notebook.is_file():
        raise FileNotFoundError("Export and upload the executed Colab notebook first")
    notebook = json.loads(args.notebook.read_text())
    if notebook.get("nbformat") != 4 or not notebook.get("cells"):
        raise ValueError("Expected a nonempty version-4 notebook")
    required = [outputs / "final_model.cbm", outputs / "final_model_metadata.json",
                outputs / "frozen_model_recipe.json", outputs / "validation_predictions.csv",
                outputs / "december_predictions.csv", project / "score.py"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if repo.exists():
        raise FileExistsError(f"Destination exists: {repo}. Choose a new --destination for a fresh build.")
    metadata = json.loads((outputs / "final_model_metadata.json").read_text())
    frozen = json.loads((outputs / "frozen_model_recipe.json").read_text())
    if metadata["model_name"] != "core_rmse" or metadata["target_scale"] != 1.0:
        raise ValueError("This package expects the selected core RMSE model with dollar outputs")
    if metadata["trees"] != 425:
        raise ValueError("The reported final tree count was 425; inspect this different artifact before packaging")
    config = {"parameters": frozen["recipes"]["chart_core_rmse"]["parameters"],
              "iterations": metadata["trees"], "target_scale": 1.0,
              "features": metadata["features"], "categorical_features": metadata["categorical_features"]}
    repo.mkdir(parents=True)
    for directory in ["data", "outputs", "artifacts", "notebooks"]:
        (repo / directory).mkdir()
    for path in (project / "data").glob("*.csv"):
        shutil.copy2(path, repo / "data" / path.name)
    for path in outputs.iterdir():
        if path.is_file() and path.suffix in {".csv", ".json", ".png"}:
            shutil.copy2(path, repo / "outputs" / path.name)
    shutil.copy2(outputs / "final_model.cbm", repo / "artifacts" / "final_model.cbm")
    shutil.copy2(outputs / "final_model_metadata.json", repo / "artifacts" / "final_model_metadata.json")
    shutil.copy2(outputs / "validation_predictions.csv", repo / "validation_predictions.csv")
    shutil.copy2(project / "score.py", repo / "score.py")
    shutil.copy2(args.notebook, repo / "notebooks" / "freight_rate_prediction.ipynb")
    shutil.copy2(Path(__file__).with_name("solution.py"), repo / "solution.py")
    (repo / "model_config.json").write_text(json.dumps(config, indent=2) + "\n")
    packages = ["numpy", "pandas", "matplotlib", "scikit-learn", "scipy", "catboost", "reportlab"]
    versions = {name: importlib.metadata.version(name) for name in packages}
    (repo / "requirements.txt").write_text("".join(f"{name}=={version}\n" for name, version in versions.items()))
    (repo / "environment.json").write_text(json.dumps({"python": platform.python_version(), "packages": versions}, indent=2) + "\n")
    (repo / "README.md").write_text(README)
    (repo / ".gitignore").write_text(".venv/\n__pycache__/\n.ipynb_checkpoints/\n*.pkl\n")

    commands = [
        [sys.executable, str(repo / "solution.py"), "--root", str(repo), "--mode", "verify"],
        [sys.executable, str(repo / "score.py"), "--predictions", str(repo / "validation_predictions.csv"),
         "--december-predictions", str(repo / "outputs" / "december_predictions.csv"),
         "--output-dir", str(repo / "scorer_results")],
    ]
    logs = []
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode:
            print(result.stderr)
            raise RuntimeError("Verification failed; no submission archive produced")
        logs.append(result.stdout)
    (repo / "outputs" / "packaging_verification.txt").write_text("\n".join(logs))
    recorded = development_metrics(repo / "outputs")
    recorded.to_csv(repo / "outputs" / "report_development_metrics.csv", index=False)
    make_report(repo, metadata, config, recorded)
    # Include the report builder so the report can be rebuilt from archived artifacts.
    shutil.copy2(Path(__file__), repo / "build_submission.py")
    archive = shutil.make_archive(str(repo), "zip", root_dir=repo.parent, base_dir=repo.name)
    print(f"REPORT: {repo / 'freight_rate_report.pdf'}")
    print(f"SUBMISSION: {repo / 'validation_predictions.csv'}")
    print(f"REPOSITORY ARCHIVE: {archive}")
    print("Saved-model reproduction and official scorer checks both passed.")


if __name__ == "__main__":
    main()
