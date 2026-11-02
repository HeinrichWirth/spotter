"""Causal route residuals and observed US federal holiday effects."""

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


def days(dates):
    return pd.to_datetime(dates).to_numpy().astype("datetime64[D]").astype(int)


def holidays(loads):
    dates = pd.to_datetime(loads.date)
    calendar = USFederalHolidayCalendar().holidays(
        f"{dates.dt.year.min() - 1}-01-01", f"{dates.dt.year.max() + 1}-12-31"
    )
    difference = days(dates)[:, None] - days(calendar)[None, :]
    before = np.min(np.where(difference <= 0, -difference, 10000), axis=1)
    after = np.min(np.where(difference >= 0, difference, 10000), axis=1)
    return np.c_[
        np.exp(-before / 3),
        np.exp(-after / 3),
        (difference == 0).any(axis=1).astype(float),
    ]


class RouteHistory:
    """Store capped out-of-time residuals by route, optionally split by equipment."""

    def __init__(self, history, training):
        self.cutoff = int(days(training.date).max() + 1)
        # A shared history can span several folds. Store only rows belonging to this prefix.
        past = history.loc[history.load_id.isin(training.load_id)].copy()
        if (
            len(past)
            and not (
                pd.to_datetime(past.training_last_date) < pd.to_datetime(past.date)
            ).all()
        ):
            raise ValueError(
                "Route history requires predictions made before the load date"
            )
        past["day"] = days(past.date)
        past["error"] = np.clip(past.actual - past.prediction, -60, 60) / past.distance
        self.tables = {}
        for fields in [("pickup", "delivery"), ("pickup", "delivery", "equipment")]:
            totals = past.groupby([*fields, "day"], sort=True).error.agg(
                ["sum", "count"]
            )
            groups = {}
            for key, group in totals.groupby(
                level=list(range(len(fields))), sort=False
            ):
                groups[key] = (
                    group.index.get_level_values("day").to_numpy(),
                    np.r_[0, group["sum"].to_numpy().cumsum()],
                    np.r_[0, group["count"].to_numpy().cumsum()],
                )
            self.tables[fields] = groups

    def transform(self, loads, fit=False):
        query = np.minimum(days(loads.date), self.cutoff)
        blocks = []
        for fields, table in self.tables.items():
            prior = np.zeros(len(loads))
            for key, rows in loads.groupby(list(fields), sort=False).indices.items():
                if key in table:
                    dates, residuals, counts = table[key]
                    index = np.searchsorted(dates, query[rows], side="left")
                    prior[rows] = residuals[index] / (counts[index] + 20)
            blocks.append(prior)
        calendar = holidays(loads)
        blocks.extend(
            [
                calendar * loads.equipment.eq(name).to_numpy()[:, None]
                for name in ["Reefer", "Flatbed"]
            ]
        )
        values = np.column_stack(blocks)
        if fit:
            self.center = values.mean(axis=0)
        return values - self.center
