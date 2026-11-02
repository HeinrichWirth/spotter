"""Calendar forecast of the market index, fitted to observed training dates."""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


class MarketReconstruction:
    @staticmethod
    def _days(loads):
        return (
            pd.to_datetime(loads.date) - pd.Timestamp("2025-01-01")
        ).dt.days.to_numpy()

    @staticmethod
    def _basis(days):
        return np.c_[
            np.ones(len(days)),
            np.sin(2 * np.pi * days / 365),
            np.cos(2 * np.pi * days / 365),
            np.sin(2 * np.pi * days / 7),
            np.cos(2 * np.pi * days / 7),
        ]

    def _design(self, days):
        basis = self._basis(days)
        if self.break_days.size:
            basis = np.c_[
                basis, np.column_stack([days >= day for day in self.break_days])
            ]
        return basis

    def fit(self, loads):
        daily = loads.groupby("date", as_index=False).market_index.mean().dropna()
        days = self._days(daily)
        market = daily.market_index.to_numpy()
        differences = np.diff(self._basis(days)[:, 1:], axis=0)
        changes = np.diff(market)
        fit = least_squares(
            lambda b: differences @ b - changes,
            x0=np.linalg.lstsq(differences, changes, rcond=None)[0],
            loss="soft_l1",
            f_scale=0.006,
        )
        self.break_days = days[1:][abs(changes - differences @ fit.x) > 0.04]
        self.coefficients = np.linalg.lstsq(self._design(days), market, rcond=None)[0]
        return self

    def predict(self, loads):
        return self._design(self._days(loads)) @ self.coefficients
