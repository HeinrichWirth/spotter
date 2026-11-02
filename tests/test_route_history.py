import unittest

import numpy as np
import pandas as pd

from freight.history import RouteHistory


class RouteHistoryTests(unittest.TestCase):
    def setUp(self):
        self.history = pd.DataFrame(
            {
                "load_id": ["a", "b", "c", "excluded", "future"],
                "date": [
                    "2025-02-01",
                    "2025-02-02",
                    "2025-02-02",
                    "2025-02-01",
                    "2025-02-03",
                ],
                "pickup": "Lexington",
                "delivery": "Fort Wayne",
                "equipment": ["Dry Van", "Dry Van", "Reefer", "Dry Van", "Dry Van"],
                "distance": 100,
                "actual": [150, 120, 130, 150, 150],
                "prediction": 100,
                "training_last_date": "2025-01-31",
            }
        )
        self.training = self.history.iloc[:3].copy()

    def test_priors_only_use_earlier_days_in_the_training_fold(self):
        routes = RouteHistory(self.history, self.training)
        query = self.history.iloc[[0, 1, 0, 0]].copy()
        query["date"] = ["2025-02-01", "2025-02-02", "2025-02-03", "2025-03-01"]
        # The first day has zero prior, so centering on it exposes raw route values.
        routes.transform(query.iloc[:1], fit=True)
        priors = routes.transform(query)[:, :2]
        expected = [
            [0, 0],
            [0.5 / 21, 0.5 / 21],
            [1 / 23, 0.7 / 22],
            [1 / 23, 0.7 / 22],
        ]
        np.testing.assert_allclose(priors, expected, rtol=0, atol=1e-12)

    def test_rejects_history_fitted_on_or_after_the_load_date(self):
        for last_date in ["2025-02-01", "2025-02-02"]:
            with self.subTest(last_date=last_date):
                history = self.history.copy()
                history.loc[0, "training_last_date"] = last_date
                with self.assertRaisesRegex(ValueError, "before the load date"):
                    RouteHistory(history, self.training)
