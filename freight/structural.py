"""Robust dollar regression for histories covering less than two quarters."""

import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.preprocessing import SplineTransformer


class TrendFare:
    def features(self, loads, fit=False):
        miles = loads.distance.to_numpy(dtype=float)
        weight = loads.weight.abs()
        market = loads.get("market_index", pd.Series(np.nan, index=loads.index))
        if fit:
            self.weight_median = float(weight.median())
            self.market_median = float(market.median())
            self.cities = sorted(set(loads.pickup) | set(loads.delivery))
            self.spline = SplineTransformer(
                n_knots=7,
                degree=3,
                knots="quantile",
                include_bias=False,
                extrapolation="linear",
            ).fit(np.log(miles)[:, None])
        dates = pd.to_datetime(loads.date)
        days = dates.dt.dayofyear.to_numpy() - 1
        quarter = dates.dt.to_period("Q")
        progress = (
            (dates - quarter.dt.start_time).dt.days
            / ((quarter + 1).dt.start_time - quarter.dt.start_time).dt.days
        ).to_numpy()
        columns = [
            np.ones(len(loads)),
            1 / miles,
            np.tanh((weight.fillna(self.weight_median).to_numpy() - 30000) / 8000),
            weight.isna().to_numpy(),
            (loads.weight < 0).to_numpy(),
            market.fillna(self.market_median).to_numpy() - 1,
            market.isna().to_numpy(),
            np.sin(2 * np.pi * days / 7),
            np.cos(2 * np.pi * days / 7),
            progress,
        ]
        for name in ["Reefer", "Flatbed"]:
            flag = loads.equipment.eq(name).to_numpy()
            columns.extend([flag, flag / miles])
        columns.append(self.spline.transform(np.log(miles)[:, None]))
        origin = loads.pickup.to_numpy()
        destination = loads.delivery.to_numpy()
        counts = np.column_stack(
            [
                (origin == city).astype(float) + (destination == city).astype(float)
                for city in self.cities
            ]
        )
        columns.append(counts)
        design = np.column_stack(columns) * miles[:, None]
        if not fit:
            missing = (~loads.pickup.isin(self.cities)).astype(int) + (
                ~loads.delivery.isin(self.cities)
            ).astype(int)
            design[:, -len(self.cities) :] += (
                miles[:, None] * missing.to_numpy()[:, None] / len(self.cities)
            )
        elapsed = (dates - pd.Timestamp("2025-01-01")).dt.days.to_numpy() / 365
        return np.c_[design, miles * elapsed]

    def fit(self, loads):
        raw = self.features(loads, fit=True)
        self.scale = np.maximum(np.sqrt(np.mean(raw * raw, axis=0)), 1e-8)
        x = raw / self.scale
        price = loads.posted_rate.to_numpy()
        ridge = np.eye(x.shape[1]) * 10
        weights = np.ones(len(loads))
        coefficients = np.zeros(x.shape[1])
        for _ in range(35):
            update = solve(
                x.T @ (x * weights[:, None]) + ridge,
                x.T @ (weights * price),
                assume_a="pos",
            )
            weights = np.minimum(1, 60 / np.maximum(abs(price - x @ update), 1e-10))
            change = np.max(abs(update - coefficients))
            coefficients = update
            if change < 1e-5:
                break
        self.coefficients = coefficients
        return self

    def predict(self, loads):
        return np.maximum(0.01, self.features(loads) @ (self.coefficients / self.scale))
