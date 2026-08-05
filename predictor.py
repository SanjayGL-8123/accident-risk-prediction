"""
predictor.py
------------
Loads the trained accident risk model and exposes a predict() function that
takes a road segment's data and returns:
    - a high-risk probability
    - a risk level (Low / Medium / High)
    - a list of specific risk factors flagged for that segment
"""

import joblib
import numpy as np

MODEL_PATH = "model/accident_risk_model.joblib"

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def get_feature_importances():
    """Return the model's global feature importances, sorted descending."""
    bundle = _load()
    importances = bundle["feature_importances"]
    return sorted(importances.items(), key=lambda x: -x[1])


def _risk_level(probability):
    if probability >= 0.6:
        return "High"
    elif probability >= 0.35:
        return "Medium"
    else:
        return "Low"


def _flag_risk_factors(segment):
    """Simple, transparent rule-based flags to explain the score to a traffic planner."""
    flags = []

    if segment["speed_limit_kmh"] >= 80:
        flags.append("High speed limit (80+ km/h)")
    if segment["traffic_volume_per_day"] >= 25000:
        flags.append("Very high daily traffic volume")
    if segment["number_of_intersections"] >= 5:
        flags.append("High number of intersections along the segment")
    if segment["pedestrian_crossings"] >= 3:
        flags.append("Multiple pedestrian crossings increase conflict points")
    if segment["road_curvature_score"] >= 7:
        flags.append("High road curvature (sharp bends)")
    if segment["road_condition_score"] <= 3:
        flags.append("Poor road surface condition")
    if segment["has_streetlights"] == 0:
        flags.append("No streetlights (reduced night visibility)")
    if segment["rainy_days_per_month"] >= 15:
        flags.append("Frequent rain (reduced traction/visibility)")
    if segment["historical_accidents_last_year"] >= 5:
        flags.append("High number of accidents recorded in the past year")

    return flags


def predict(segment):
    """
    segment: dict with keys matching FEATURE_NAMES in dataset.py:
        traffic_volume_per_day, speed_limit_kmh, number_of_lanes,
        has_streetlights, number_of_intersections, pedestrian_crossings,
        road_curvature_score, road_condition_score, rainy_days_per_month,
        historical_accidents_last_year
    """
    from dataset import FEATURE_NAMES

    bundle = _load()
    model = bundle["model"]

    X = np.array([[segment[name] for name in FEATURE_NAMES]])
    probability = model.predict_proba(X)[0][1]  # probability of class 1 (high risk)

    return {
        "risk_probability": round(probability * 100, 1),
        "risk_level": _risk_level(probability),
        "risk_factors": _flag_risk_factors(segment),
    }
