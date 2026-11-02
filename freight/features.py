"""Shipment features learned from each training prefix."""

import numpy as np
from sklearn.preprocessing import SplineTransformer


COORDINATES = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]


class ShipmentFeatures:
    def _geometry(self, loads, fit):
        coords = loads[COORDINATES].astype(float)
        if fit:
            self.coordinate_fill = coords.median()
        lat1, lon1, lat2, lon2 = np.radians(
            coords.fillna(self.coordinate_fill).to_numpy()
        ).T
        delta_lat, delta_lon = lat2 - lat1, lon2 - lon1
        haversine = (
            np.sin(delta_lat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
        )
        miles = 3958.7613 * 2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
        bearing = np.arctan2(
            np.sin(delta_lon) * np.cos(lat2),
            np.cos(lat1) * np.sin(lat2)
            - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon),
        )
        return np.c_[
            np.log1p(miles),
            np.log(loads.distance.to_numpy() / np.maximum(miles, 1)),
            np.sin(bearing),
            np.cos(bearing),
            (lat1 + lat2) / 2,
            (lon1 + lon2) / 2,
            delta_lat,
            delta_lon,
        ]

    def _cargo(self, loads, fit):
        weight = loads.weight.abs()
        log_distance = np.log(loads.distance.to_numpy(dtype=float))
        if fit:
            self.weight_fill = float(weight.median())
            self.weight_bounds = weight.quantile([0.005, 0.995]).to_numpy()
            self.distance_bounds = np.quantile(log_distance, [0.005, 0.995])
            self.distance_median = np.median(log_distance)
            self.distance_iqr = max(
                float(
                    np.quantile(log_distance, 0.75) - np.quantile(log_distance, 0.25)
                ),
                1e-6,
            )
        bounded_weight = np.clip(
            weight.fillna(self.weight_fill).to_numpy(), *self.weight_bounds
        )
        bounded_distance = np.clip(log_distance, *self.distance_bounds)
        if fit:
            self.weight_spline = SplineTransformer(
                n_knots=5,
                degree=3,
                knots="quantile",
                include_bias=False,
                extrapolation="constant",
            ).fit(bounded_weight[:, None])
            self.distance_spline = SplineTransformer(
                n_knots=7,
                degree=3,
                knots="quantile",
                include_bias=False,
                extrapolation="constant",
            ).fit(bounded_distance[:, None])
        weight_shape = self.weight_spline.transform(bounded_weight[:, None])
        distance_shape = self.distance_spline.transform(bounded_distance[:, None])
        if fit:
            self.weight_center = weight_shape.mean(axis=0)
            self.distance_center = distance_shape.mean(axis=0)
        weight_shape -= self.weight_center
        distance_shape -= self.distance_center
        equipment = [
            loads.equipment.eq(name).to_numpy()[:, None]
            for name in ["Reefer", "Flatbed"]
        ]
        weight_level = np.tanh((bounded_weight - 30000.0) / 8000.0)
        mileage_level = np.clip(
            (log_distance - self.distance_median) / self.distance_iqr, -3.0, 3.0
        )
        return np.column_stack(
            [
                weight_shape,
                *[weight_shape * flag for flag in equipment],
                *[distance_shape * flag for flag in equipment],
                weight_level * mileage_level,
                weight_level * mileage_level**2,
            ]
        )

    def _direction(self, loads, fit):
        roles = ["pickup", "delivery"]
        if fit:
            cities = sorted(set(loads.pickup) | set(loads.delivery))
            self.city_index = {city: i for i, city in enumerate(cities)}
            self.role_frequency = {
                role: loads[role]
                .value_counts(normalize=True)
                .reindex(cities[:-1], fill_value=0)
                .to_numpy()
                for role in roles
            }
        endpoints = []
        for role in roles:
            frequencies = self.role_frequency[role]
            contrast = np.tile(-frequencies, (len(loads), 1))
            index = loads[role].map(self.city_index).fillna(-1).to_numpy(dtype=int)
            active = (index >= 0) & (index < len(frequencies))
            contrast[np.flatnonzero(active), index[active]] += 1
            contrast[index < 0] = 0
            endpoints.append(contrast)
        return endpoints[0] - endpoints[1]

    def transform(self, loads, fit=False):
        blocks = [
            self._geometry(loads, fit),
            self._cargo(loads, fit),
            self._direction(loads, fit),
        ]
        if fit:
            self.centers = [block.mean(axis=0) for block in blocks]
        return np.column_stack(
            [block - center for block, center in zip(blocks, self.centers)]
        )
