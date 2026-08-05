"""
dataset.py
----------
Generates a synthetic (but realistically structured) road-segment dataset
for the Accident Risk Prediction project.

Each road segment has 10 features commonly available to traffic authorities:
    - traffic_volume_per_day        (vehicles/day, 500-50000)
    - speed_limit_kmh                (30-120)
    - number_of_lanes                (1-6)
    - has_streetlights               (1 = yes, 0 = no)
    - number_of_intersections        (0-10, along the segment)
    - pedestrian_crossings           (0-5)
    - road_curvature_score           (0-10, higher = more bends/curves)
    - road_condition_score           (0-10, higher = better surface condition)
    - rainy_days_per_month           (0-30)
    - historical_accidents_last_year (0-20, prior recorded accidents)

The label `high_risk` (1 = high accident risk, 0 = low risk) is generated
from a weighted "true risk" formula plus random noise, mirroring real traffic
safety research (poor lighting, high speed, curvature, intersections, and
past accident history are all known risk factors).

NOTE: This is a synthetic demo dataset for building and testing the full
pipeline. For real deployment, replace this with actual traffic/accident
records from your transportation authority.
"""

import numpy as np


def generate_dataset(n_samples=800, random_state=42):
    rng = np.random.default_rng(random_state)

    traffic_volume_per_day = np.clip(rng.normal(15000, 9000, n_samples), 500, 50000)
    speed_limit_kmh = np.clip(rng.normal(65, 25, n_samples), 30, 120)
    number_of_lanes = np.clip(rng.integers(1, 7, n_samples), 1, 6)
    has_streetlights = rng.binomial(1, 0.55, n_samples)
    number_of_intersections = np.clip(rng.poisson(3, n_samples), 0, 10)
    pedestrian_crossings = np.clip(rng.poisson(1.2, n_samples), 0, 5)
    road_curvature_score = np.clip(rng.normal(4, 2.5, n_samples), 0, 10)
    road_condition_score = np.clip(rng.normal(6, 2, n_samples), 0, 10)
    rainy_days_per_month = np.clip(rng.normal(8, 6, n_samples), 0, 30)
    historical_accidents_last_year = np.clip(rng.poisson(2.5, n_samples), 0, 20)

    # Weighted "true risk" score: higher = more accident-prone.
    # Coefficients reflect commonly cited traffic safety factors.
    risk_score = (
        0.00006 * traffic_volume_per_day
        + 0.022 * speed_limit_kmh
        + 0.18 * number_of_intersections
        + 0.15 * pedestrian_crossings
        + 0.22 * road_curvature_score
        - 0.20 * road_condition_score
        - 0.90 * has_streetlights
        + 0.04 * rainy_days_per_month
        + 0.22 * historical_accidents_last_year
        - 4.0  # baseline offset (tuned for a realistic ~35% high-risk rate)
    )

    noise = rng.normal(0, 1.2, n_samples)
    probability = 1 / (1 + np.exp(-(risk_score + noise)))  # logistic squashing
    high_risk = rng.binomial(1, probability)

    dataset = {
        "traffic_volume_per_day": np.round(traffic_volume_per_day, 0),
        "speed_limit_kmh": np.round(speed_limit_kmh, 0),
        "number_of_lanes": number_of_lanes.astype(int),
        "has_streetlights": has_streetlights.astype(int),
        "number_of_intersections": number_of_intersections.astype(int),
        "pedestrian_crossings": pedestrian_crossings.astype(int),
        "road_curvature_score": np.round(road_curvature_score, 1),
        "road_condition_score": np.round(road_condition_score, 1),
        "rainy_days_per_month": np.round(rainy_days_per_month, 1),
        "historical_accidents_last_year": historical_accidents_last_year.astype(int),
        "high_risk": high_risk.astype(int),
    }

    return dataset


FEATURE_NAMES = [
    "traffic_volume_per_day",
    "speed_limit_kmh",
    "number_of_lanes",
    "has_streetlights",
    "number_of_intersections",
    "pedestrian_crossings",
    "road_curvature_score",
    "road_condition_score",
    "rainy_days_per_month",
    "historical_accidents_last_year",
]


def to_feature_matrix(dataset):
    """Convert the dict-of-arrays dataset into an (X, y) pair for sklearn."""
    X = np.column_stack([dataset[name] for name in FEATURE_NAMES])
    y = dataset["high_risk"]
    return X, y
