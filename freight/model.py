"""Training and prediction for individual loads and the reduced-input chart."""

import numpy as np
import pandas as pd

from .features import ShipmentFeatures
from .hierarchical import HierarchicalFare
from .history import RouteHistory, holidays
from .market import MarketReconstruction
from .structural import TrendFare


CHART_FIELDS = ["pickup", "delivery", "distance", "equipment", "weight", "date"]
HISTORY_FIELDS = ["load_id", "date", "pickup", "delivery", "equipment", "distance"]


class ShipmentHierarchy(HierarchicalFare):
    def __init__(self, routes):
        self.shipment = ShipmentFeatures()
        self.routes = routes

    def _static(self, loads, market, missing, fit=False):
        base = super()._static(loads, market, missing, fit)
        blocks = [base, self.shipment.transform(loads, fit)]
        if self.routes is not None:
            blocks.append(self.routes.transform(loads, fit))
        return np.column_stack(blocks)

    def _calendar(self, loads, fit=False):
        base, quarter = super()._calendar(loads, fit)
        dates = pd.to_datetime(loads.date)
        next_quarter = (dates.dt.to_period("Q") + 1).dt.start_time
        remaining = (next_quarter - dates).dt.days.to_numpy() - 1
        extras = [quarter, np.exp(-remaining / 12.0), np.exp(-remaining / 30.0)]
        if self.routes is not None:
            extras.append(holidays(loads))
        return base, np.column_stack(extras)


class ShipmentTrend(TrendFare):
    def __init__(self, routes):
        self.shipment = ShipmentFeatures()
        self.routes = routes

    def features(self, loads, fit=False):
        base = super().features(loads, fit)
        extras = [self.shipment.transform(loads, fit)]
        if self.routes is not None:
            extras.extend([self.routes.transform(loads, fit), holidays(loads)])
        return np.c_[base, np.column_stack(extras) * loads.distance.to_numpy()[:, None]]


def _fit_estimator(loads, chart, routes):
    dates = pd.to_datetime(loads.date)
    support = dates.groupby(dates.dt.to_period("Q")).nunique()
    if (support >= 28).sum() >= 2:
        model = HierarchicalFare() if chart else ShipmentHierarchy(routes)
        return model.fit(loads), None
    model = TrendFare() if chart else ShipmentTrend(routes)
    return model.fit(loads), MarketReconstruction().fit(loads)


def _predict(model, market, loads):
    if market is not None:
        loads = loads.copy()
        fallback = pd.Series(market.predict(loads), index=loads.index)
        loads["market_index"] = (
            loads.market_index.fillna(fallback) if "market_index" in loads else fallback
        )
    return model.predict(loads)


def build_route_history(training):
    """Predict each month using only preceding months. The first month is a warm-up."""
    months = pd.to_datetime(training.date).dt.to_period("M")
    records = []
    for month in sorted(months.unique())[1:]:
        past = training.loc[months < month]
        current = training.loc[months == month]
        model, market = _fit_estimator(past, chart=False, routes=None)
        row = current[HISTORY_FIELDS].copy()
        row["actual"] = current.posted_rate
        row["prediction"] = _predict(model, market, current.drop(columns="posted_rate"))
        row["training_last_date"] = past.date.max()
        records.append(row)
    return (
        pd.concat(records, ignore_index=True)
        if records
        else pd.DataFrame(
            columns=[*HISTORY_FIELDS, "actual", "prediction", "training_last_date"]
        )
    )


class FreightRateModel:
    """Predict a total load price in USD. Pass chart=True for the six-input scenario.

    A precomputed route history from build_route_history can be shared across
    expanding training windows. A standalone fit builds its own history.
    """

    def __init__(self, chart=False, history=None):
        self.chart = chart
        self.history = history

    def fit(self, training):
        routes = None
        if not self.chart:
            history = (
                self.history
                if self.history is not None
                else build_route_history(training)
            )
            routes = RouteHistory(history, training)
        self.model, self.market = _fit_estimator(training, self.chart, routes)
        self.history = None
        return self

    def predict(self, loads):
        if "posted_rate" in loads:
            raise ValueError("Remove posted_rate from prediction inputs")
        if self.chart:
            loads = loads[CHART_FIELDS]
        return _predict(self.model, self.market, loads)
