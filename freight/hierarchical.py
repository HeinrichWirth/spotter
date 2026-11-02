"""Decompose rates into shipment characteristics and a daily market level."""

import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.preprocessing import SplineTransformer

from .market import MarketReconstruction


class HierarchicalFare:
    def _market(self, loads, fit=False):
        raw = loads.get("market_index", pd.Series(np.nan, index=loads.index))
        if fit:
            self.market_median = float(raw.median())
            daily = loads.groupby("date").market_index.mean()
            fallback = loads.date.map(daily).fillna(self.market_median)
        else:
            fallback = pd.Series(self.market_forecast.predict(loads), index=loads.index)
        return raw.fillna(fallback).to_numpy(), raw.isna().to_numpy()

    def _static(self, loads, market, missing_market, fit=False):
        miles = loads.distance.to_numpy(dtype=float)
        weight = loads.weight.abs()
        if fit:
            self.weight_median = float(weight.median())
            cities = sorted(set(loads.pickup) | set(loads.delivery))
            self.city_index = {city: i for i, city in enumerate(cities)}
            counts = pd.concat([loads.pickup, loads.delivery]).value_counts()
            self.city_frequency = counts.reindex(cities).to_numpy() / (2 * len(loads))
            self.distance_spline = SplineTransformer(
                n_knots=7,
                degree=3,
                knots="quantile",
                include_bias=False,
                extrapolation="linear",
            ).fit(np.log(miles)[:, None])
        columns = [
            1 / miles,
            np.tanh((weight.fillna(self.weight_median).to_numpy() - 30000) / 8000),
            weight.isna().to_numpy(),
            (loads.weight < 0).to_numpy(),
            market - 1,
            missing_market,
        ]
        for equipment in ["Reefer", "Flatbed"]:
            flag = loads.equipment.eq(equipment).to_numpy()
            columns.extend([flag, flag / miles])
        columns.append(self.distance_spline.transform(np.log(miles)[:, None]))
        # Both endpoints share coefficients. Unseen cities use the training mean.
        frequencies = self.city_frequency[:-1]
        city = np.tile(-2 * frequencies, (len(loads), 1))
        for role in ["pickup", "delivery"]:
            idx = loads[role].map(self.city_index).fillna(-1).to_numpy(dtype=int)
            active = (idx >= 0) & (idx < len(frequencies))
            city[np.flatnonzero(active), idx[active]] += 1
            city[idx < 0] += frequencies
        columns.append(city)
        return np.column_stack(columns).astype(float)

    def _calendar(self, loads, fit=False):
        dates = pd.to_datetime(loads.date)
        days = (dates - pd.Timestamp("2025-01-01")).dt.days.to_numpy()
        base = np.column_stack(
            [np.ones(len(loads)), days / 365]
            + [(dates.dt.dayofweek.to_numpy() == i).astype(float) for i in range(1, 7)]
        )
        quarter = dates.dt.to_period("Q")
        progress = (
            (dates - quarter.dt.start_time).dt.days
            / ((quarter + 1).dt.start_time - quarter.dt.start_time).dt.days
        ).to_numpy()
        if fit:
            self.quarter_spline = SplineTransformer(
                n_knots=5,
                degree=3,
                knots=np.linspace(0, 1, 5)[:, None],
                include_bias=False,
                extrapolation="linear",
            ).fit(progress[:, None])
        return base, self.quarter_spline.transform(progress[:, None])

    def fit(self, loads):
        self.market_forecast = MarketReconstruction().fit(loads)
        market, missing = self._market(loads, fit=True)
        raw = self._static(loads, market, missing, fit=True)
        self.scale = np.maximum(np.sqrt(np.mean(raw * raw, axis=0)), 1e-6)
        design = raw / self.scale
        miles = loads.distance.to_numpy()
        rate_per_mile = loads.posted_rate.to_numpy() / miles
        dates, day = np.unique(loads.date, return_inverse=True)
        base_weight = miles**2 / np.mean(miles**2)
        robust = np.ones(len(loads))
        coefficients = np.zeros(design.shape[1])
        effects = np.zeros(len(dates))
        penalty = 10 * np.eye(design.shape[1])

        # Demeaning within dates isolates structural effects from the daily rate level.
        for _ in range(45):
            weight = base_weight * robust
            total = np.bincount(day, weights=weight, minlength=len(dates))
            avg_x = np.column_stack(
                [
                    np.bincount(day, weights=weight * col, minlength=len(dates)) / total
                    for col in design.T
                ]
            )
            avg_y = (
                np.bincount(day, weights=weight * rate_per_mile, minlength=len(dates))
                / total
            )
            x = design - avg_x[day]
            y = rate_per_mile - avg_y[day]
            update = solve(
                x.T @ (weight[:, None] * x) + penalty,
                x.T @ (weight * y),
                assume_a="pos",
            )
            daily_update = avg_y - avg_x @ update
            error = miles * (rate_per_mile - design @ update - daily_update[day])
            robust = np.minimum(1, 60 / np.maximum(abs(error), 1e-12))
            change = max(
                np.max(abs(update - coefficients)), np.max(abs(daily_update - effects))
            )
            coefficients, effects = update, daily_update
            if change < 1e-7:
                break
        self.coefficients = coefficients
        daily = pd.DataFrame({"date": dates})
        daily_market = (
            loads.groupby("date")
            .market_index.mean()
            .reindex(dates)
            .fillna(self.market_median)
        )
        calendar, quarter = self._calendar(daily, fit=True)
        extra = np.c_[quarter, daily_market.to_numpy() - 1]

        # Keep the trend and weekdays unpenalized. Penalize residual calendar and market effects.
        weight = total / total.mean()
        gram = calendar.T @ (weight[:, None] * calendar)
        self.projection = np.linalg.solve(gram, calendar.T @ (weight[:, None] * extra))
        self.time_coefficients = np.linalg.solve(gram, calendar.T @ (weight * effects))
        orthogonal = extra - calendar @ self.projection
        target = effects - calendar @ self.time_coefficients
        active = np.sqrt(np.sum(weight[:, None] * orthogonal**2, axis=0)) > 1e-8
        penalties = np.array([0.1] * (extra.shape[1] - 1) + [0.05])
        x = orthogonal[:, active]
        self.extra_coefficients = np.zeros(extra.shape[1])
        self.extra_coefficients[active] = solve(
            x.T @ (weight[:, None] * x) + np.diag(penalties[active]),
            x.T @ (weight * target),
            assume_a="pos",
        )
        return self

    def predict(self, loads):
        market, missing = self._market(loads)
        x = self._static(loads, market, missing)
        base, quarter = self._calendar(loads)
        extra = np.c_[quarter, market - 1]
        temporal = (
            base @ self.time_coefficients
            + (extra - base @ self.projection) @ self.extra_coefficients
        )
        return np.maximum(
            0.01,
            loads.distance.to_numpy()
            * (x @ (self.coefficients / self.scale) + temporal),
        )
