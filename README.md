# Spotter | Freight rate prediction

This model estimates the total posted price in US dollars for an individual truckload. It was trained on 48,000 shipments from January through October 2025 and used to price 12,000 shipments in November and December.

| Submission | Contents |
| --- | --- |
| [Assessment report](report/Freight_Rate_Assessment.pdf) | Data quality, model selection, monthly accuracy, extreme prices and the December chart |
| [Shipment predictions](validation_predictions.csv) | One predicted rate for every load, in the original template order |
| [December scenario](december_predictions.csv) and [chart](scorer_results/candidate_december.png) | Daily quotes for one Lexington to Fort Wayne dry van shipment, with only the date changing |
| [Monthly results](evaluation/monthly_metrics.csv) | Load counts, mean and median absolute errors, and the retrospective price-ratio diagnostic |

## What the historical forecasts showed

Mean monthly absolute error was **$89.35**, with monthly values from **$77.10 in April** to **$106.36 in June**. In each of the seven months, at least half of quotes were within $29 of the posted price. The 338 largest misses accounted for 55.46% of total absolute error. Comparable loads on the same routes often had ordinary prices, and the available shipment fields did not reliably identify those extremes in advance. Every load counts toward the reported accuracy.

The model separates the effects of shipment characteristics from a daily rate level. It learns the shipment effects within dates using Huber loss, then forecasts the daily level using calendar and market information. Accuracy improved when the model accounted for proximity to US federal holidays and prior forecast errors on the same directed route. A simpler trend model handles short training histories.

## Training and validation

| Training period | Forecast period |
| --- | --- |
| January to March | April and May |
| January to May | June and July |
| January to June | July and August |
| January to August | September and October |

Each fit uses only earlier dates. Each historical route error also comes from a prediction made using earlier months. July occurs in two windows, so its reported scores use the later fit, trained through June. This leaves 33,709 distinct evaluation loads. The headline score gives each month equal weight. These same windows informed feature selection. Actual November and December prices are withheld.

## Reproduce the submission

The run uses Python 3.12 and a CPU. Clone this repository, then create a virtual environment from its root.

```shell
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS and Linux. Install the pinned dependencies.

```shell
python -m pip install -r requirements.txt
```

Put the four original assessment CSVs in a single directory. The runner accepts filenames with either hyphens or underscores and verifies their contents against [the input checksums](data.sha256).

```text
data/
  train_test.csv
  validation.csv
  validation_predictions_template.csv
  december_chart_inputs.csv
```

From the repository root, run the following. Use `--data-dir` to point to another input directory if needed.

```shell
python run.py --data-dir data --output-dir build
```

This command trains and evaluates the historical models, refits on all labeled shipments, generates both prediction files and invokes the supplied scorer. The files under `build/` mirror the published predictions, `evaluation/monthly_metrics.csv` and `scorer_results/candidate_december.png`. To run just the historical evaluation or final prediction, use `python run.py evaluate` or `python run.py predict` with the same directory options.

To check the published files and redraw the chart without retraining or the original data, run the supplied scorer directly.

```shell
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv --output-dir build/scorer_results
```

The scorer checks identifiers, positive prices and the fixed scenario inputs. Historical forecast errors are calculated by [the evaluation runner](run.py).

Run `python -m unittest discover -s tests -v` to check the time boundary on route history. The tests cover same-day loads, future loads and the training cutoff for each historical forecast.

## Code guide

| File | Role |
| --- | --- |
| [run.py](run.py) | Input checks, time splits, monthly scores, full-data refit and submission generation |
| [freight/model.py](freight/model.py) | Model selection by history length, chronological route forecasts and the six-input chart contract |
| [freight/hierarchical.py](freight/hierarchical.py) | Within-date shipment estimates and daily rate forecasting |
| [freight/features.py](freight/features.py) and [freight/history.py](freight/history.py) | Route geometry, shipment features, lagged route corrections and holiday timing |
| [freight/market.py](freight/market.py) and [freight/structural.py](freight/structural.py) | Market-index reconstruction and the short-history model |
