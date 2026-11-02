"""Train the freight rate models, evaluate historical forecasts and score the deliverables."""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd

from freight import FreightRateModel, build_route_history


ROOT = Path(__file__).resolve().parent
WINDOWS = (
    ("2025-04-01", "2025-06-01"),
    ("2025-06-01", "2025-08-01"),
    ("2025-07-01", "2025-09-01"),
    ("2025-09-01", "2025-11-01"),
)
SCENARIO_FIELDS = ["pickup", "delivery", "distance", "equipment", "weight", "date"]


def assessment_file(directory: Path, name: str) -> Path:
    for filename in (name, name.replace("-", "_")):
        path = directory / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing assessment input {name} in {directory}")


def read_input(directory: Path, name: str) -> pd.DataFrame:
    path = assessment_file(directory, name)
    hashes = dict(
        line.split(maxsplit=1)[::-1]
        for line in (ROOT / "data.sha256").read_text().splitlines()
    )
    digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if digest != hashes[name]:
        raise ValueError(f"{path.name} differs from the original assessment input.")
    return pd.read_csv(path)


def evaluate(training, history, output: Path):
    forecasts = []
    for start, stop in WINDOWS:
        past = training[training.date < start]
        future = training[(training.date >= start) & (training.date < stop)]
        print(f"Evaluating {start} to {stop} (exclusive)", flush=True)
        model = FreightRateModel(history=history[history.date < start]).fit(past)
        forecast = future[["load_id", "date", "posted_rate"]].copy()
        forecast["predicted_rate"] = model.predict(future.drop(columns="posted_rate"))
        forecasts.append(forecast)

    rows = pd.concat(forecasts).drop_duplicates("load_id", keep="last")
    if len(rows) != 33_709:
        raise ValueError("Unexpected number of distinct evaluation loads.")
    rows["absolute_error"] = (rows.predicted_rate - rows.posted_rate).abs()
    ratio = rows.posted_rate / rows.predicted_rate
    central = rows[ratio.between(0.5, 2)]
    monthly = rows.groupby(rows.date.str[:7]).absolute_error.agg(
        loads="size", mae="mean", median_absolute_error="median"
    )
    monthly["mae_within_factor_two"] = central.groupby(
        central.date.str[:7]
    ).absolute_error.mean()
    output.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(
        output / "monthly_metrics.csv", index_label="month", float_format="%.6f"
    )
    print(f"Mean monthly MAE: ${monthly.mae.mean():.2f}", flush=True)


def predict(training, history, directory: Path, output: Path):
    loads = read_input(directory, "validation.csv")
    template = read_input(directory, "validation-predictions-template.csv")
    scenario = read_input(directory, "december-chart-inputs.csv")
    if (
        not loads.load_id.is_unique
        or not template.load_id.is_unique
        or set(loads.load_id) != set(template.load_id)
    ):
        raise ValueError("Validation loads and template IDs must match one to one.")

    print(f"Fitting the final shipment model on {len(training):,} loads", flush=True)
    model = FreightRateModel(history=history).fit(training)
    aligned = pd.Series(model.predict(loads), index=loads.load_id).reindex(
        template.load_id
    )
    chart_model = FreightRateModel(chart=True).fit(training)
    scenario["predicted_rate"] = chart_model.predict(scenario[SCENARIO_FIELDS])
    template["predicted_rate"] = aligned.to_numpy()
    for frame in (template, scenario):
        if not (np.isfinite(frame.predicted_rate) & (frame.predicted_rate > 0)).all():
            raise ValueError("Predictions must be finite and positive.")
    output.mkdir(parents=True, exist_ok=True)
    template.to_csv(
        output / "validation_predictions.csv", index=False, float_format="%.9f"
    )
    scenario.to_csv(
        output / "december_predictions.csv", index=False, float_format="%.6f"
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "score.py"),
            "--predictions",
            str(output / "validation_predictions.csv"),
            "--december-predictions",
            str(output / "december_predictions.csv"),
            "--output-dir",
            str(output / "scorer_results"),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", nargs="?", choices=("all", "evaluate", "predict"), default="all"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    training = read_input(args.data_dir, "train-test.csv")
    print("Building route history from chronological forecasts", flush=True)
    history = build_route_history(training)
    if args.action in {"all", "evaluate"}:
        evaluate(training, history, args.output_dir / "evaluation")
    if args.action in {"all", "predict"}:
        predict(training, history, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
