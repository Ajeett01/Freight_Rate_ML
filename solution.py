"""Reproduce final freight-rate training, inference, and reserved evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

TARGET = "posted_rate"
CAT_COLUMNS = ["pickup", "delivery", "equipment", "weekday"]
CORE_COLUMNS = [
    "pickup", "delivery", "equipment", "pickup_lat", "pickup_lon",
    "delivery_lat", "delivery_lon", "distance", "weight",
    "weight_was_missing", "weight_was_invalid",
]


def input_file(data_dir, names):
    matches = [data_dir / name for name in names if (data_dir / name).is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one of {names} in {data_dir}; found {matches}")
    return matches[0]


def load_inputs(root):
    data = root / "data"
    return {
        "train": pd.read_csv(input_file(data, ["train-test.csv", "train_test.csv"])),
        "inference": pd.read_csv(input_file(data, ["validation.csv"])),
        "template": pd.read_csv(input_file(data, ["validation-predictions-template.csv", "validation_predictions_template.csv"])),
        "december": pd.read_csv(input_file(data, ["december-chart-inputs.csv", "december_chart_inputs.csv"])),
    }


def clean(frame):
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], format="%Y-%m-%d", errors="raise")
    if result["date"].isna().any():
        raise ValueError("Missing dates")
    for column in ["pickup", "delivery", "equipment"]:
        result[column] = result[column].astype("string").str.strip().fillna("__MISSING__").replace("", "__MISSING__").astype(str)
    for column in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance", "weight"]:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        if result[column].isin([np.inf, -np.inf]).any():
            raise ValueError(f"Infinite {column}")
    if result["distance"].isna().any() or result["distance"].le(0).any():
        raise ValueError("Distance must be present and positive")
    result["weight_was_missing"] = result["weight"].isna().astype("int8")
    result["weight_was_invalid"] = result["weight"].le(0).astype("int8")
    result.loc[result["weight_was_invalid"].eq(1), "weight"] = np.nan
    if TARGET in result:
        result[TARGET] = pd.to_numeric(result[TARGET], errors="raise")
        if not np.isfinite(result[TARGET]).all() or result[TARGET].le(0).any():
            raise ValueError("Invalid target")
    return result


def features(frame):
    result = frame[CORE_COLUMNS].copy()
    result["weekday"] = pd.to_datetime(frame["date"]).dt.dayofweek.astype(str)
    for column in CAT_COLUMNS:
        result[column] = result[column].fillna("__MISSING__").astype(str)
    return result


def coordinates(train):
    parts = []
    for role in ["pickup", "delivery"]:
        part = train[[role, role + "_lat", role + "_lon"]].rename(
            columns={role: "city", role + "_lat": "lat", role + "_lon": "lon"}
        )
        parts.append(part)
    result = pd.concat(parts, ignore_index=True).drop_duplicates()
    if not result["city"].is_unique:
        raise ValueError("Conflicting city coordinates")
    return result


def prepare_december(raw, city_coordinates):
    result = raw.drop(columns="predicted_rate").copy()
    lookup = city_coordinates.set_index("city")
    for role in ["pickup", "delivery"]:
        result[role + "_lat"] = result[role].map(lookup["lat"])
        result[role + "_lon"] = result[role].map(lookup["lon"])
    geo = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
    if result[geo].isna().any().any():
        raise ValueError("Unknown December scenario city")
    return clean(result)


def predict(model, frame, scale=1.0):
    values = np.asarray(model.predict(features(frame)), dtype=float) * scale
    if len(values) != len(frame) or not np.isfinite(values).all():
        raise ValueError("Invalid predictions")
    return np.maximum(values, 1.0)


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    error = np.asarray(predicted, dtype=float) - actual
    return {"MAE": float(np.abs(error).mean()), "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "WAPE_pct": float(100 * np.abs(error).sum() / np.abs(actual).sum()), "bias": float(error.mean())}


def fit_recipe(train, params, iterations=None):
    if iterations is None:
        cutoff = train["date"].max() - pd.Timedelta(days=13)
        earlier = train.loc[train["date"] < cutoff]
        inner = train.loc[train["date"] >= cutoff]
        if earlier.empty or inner.empty:
            raise ValueError("Insufficient data for inner chronological split")
        probe = CatBoostRegressor(**params, iterations=2000)
        probe.fit(features(earlier), earlier[TARGET], cat_features=CAT_COLUMNS,
                  eval_set=(features(inner), inner[TARGET]), early_stopping_rounds=100,
                  use_best_model=True, verbose=False)
        iterations = probe.get_best_iteration() + 1
    if iterations <= 0:
        raise ValueError("Invalid iteration count")
    model = CatBoostRegressor(**params, iterations=int(iterations))
    model.fit(features(train), train[TARGET], cat_features=CAT_COLUMNS, verbose=False)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--mode", choices=["predict", "verify", "train", "evaluate"], default="predict")
    parser.add_argument("--task-type", choices=["CPU", "GPU"], help="Override training hardware; predictions use CPU by default")
    parser.add_argument("--select-iterations", action="store_true", help="Repeat inner stopping in train mode instead of using the recorded final tree count")
    args = parser.parse_args()
    root = args.root.resolve()
    inputs = load_inputs(root)
    train = clean(inputs["train"])
    inference = clean(inputs["inference"])
    config = json.loads((root / "model_config.json").read_text())
    params = dict(config["parameters"])
    if args.task_type:
        params["task_type"] = args.task_type
        if args.task_type == "CPU":
            params.pop("devices", None)
    output = root / "outputs"
    artifacts = root / "artifacts"
    output.mkdir(exist_ok=True)
    artifacts.mkdir(exist_ok=True)
    if args.mode == "evaluate":
        fitting = train.loc[train["date"] <= "2025-08-31"]
        evaluation = train.loc[train["date"].between("2025-09-01", "2025-10-31")]
        model = fit_recipe(fitting, params)
        result = {"model": "reproduced_core_rmse", "trees": model.tree_count_, **metrics(evaluation[TARGET], predict(model, evaluation))}
        pd.DataFrame([result]).to_csv(output / "reproduced_reserved_metrics.csv", index=False)
        print(json.dumps(result, indent=2))
        return
    if args.mode == "train":
        model = fit_recipe(train, params, None if args.select_iterations else config["iterations"])
        model.save_model(str(artifacts / "final_model.cbm"))
        print(f"Trained {model.tree_count_} trees on {len(train):,} loads.")
    else:
        model = CatBoostRegressor()
        model.load_model(str(artifacts / "final_model.cbm"))

    city_map = coordinates(train)
    december_features = prepare_december(inputs["december"], city_map)
    lookup = pd.DataFrame({"load_id": inference["load_id"], "predicted_rate": predict(model, inference)})
    template = inputs["template"]
    required_ids = {f"TE-{i:06d}" for i in range(1, 12001)}
    if len(lookup) != 12000 or not lookup["load_id"].is_unique or set(lookup["load_id"]) != required_ids:
        raise ValueError("Inference IDs do not satisfy the submission contract")
    if len(template) != 12000 or not template["load_id"].is_unique or set(template["load_id"]) != required_ids:
        raise ValueError("Invalid template IDs")
    submission = template[["load_id"]].merge(lookup, on="load_id", how="left", sort=False, validate="one_to_one")
    december = inputs["december"].copy()
    december["predicted_rate"] = predict(model, december_features)
    if args.mode == "verify":
        checks = [(submission, root / "validation_predictions.csv", "final loads"),
                  (december, output / "december_predictions.csv", "December scenarios")]
        for generated, path, label in checks:
            saved = pd.read_csv(path)
            if list(saved.columns) != list(generated.columns) or len(saved) != len(generated):
                raise ValueError(f"Invalid saved schema for {label}")
            if label == "final loads" and saved["load_id"].tolist() != generated["load_id"].tolist():
                raise ValueError("Saved IDs are not in template order")
            np.testing.assert_allclose(saved["predicted_rate"], generated["predicted_rate"], rtol=0, atol=1e-6)
            difference = np.max(np.abs(saved["predicted_rate"] - generated["predicted_rate"]))
            print(f"Verified saved-model predictions for {len(saved):,} {label}; maximum difference ${difference:.9f}.")
        return
    submission.to_csv(root / "validation_predictions.csv", index=False, float_format="%.6f")
    december.to_csv(output / "december_predictions.csv", index=False, float_format="%.6f")
    city_map.to_csv(artifacts / "city_coordinates.csv", index=False)
    print("Created validation_predictions.csv and outputs/december_predictions.csv.")


if __name__ == "__main__":
    main()
