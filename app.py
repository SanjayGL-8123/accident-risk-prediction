"""
app.py
------
Flask web app for the Accident Risk Prediction system.

Setup:
    pip install -r requirements.txt
    python train_model.py       # trains and saves the model (run once)
    python app.py                # starts the web server

Then open http://127.0.0.1:2000
"""

import os
from flask import Flask, render_template, request
from predictor import predict, get_feature_importances, MODEL_PATH

app = Flask(__name__)

FIELD_LABELS = {
    "traffic_volume_per_day": "Traffic Volume (vehicles/day)",
    "speed_limit_kmh": "Speed Limit (km/h)",
    "number_of_lanes": "Number of Lanes",
    "has_streetlights": "Has Streetlights",
    "number_of_intersections": "Number of Intersections",
    "pedestrian_crossings": "Pedestrian Crossings",
    "road_curvature_score": "Road Curvature Score (0-10)",
    "road_condition_score": "Road Condition Score (0-10)",
    "rainy_days_per_month": "Rainy Days per Month",
    "historical_accidents_last_year": "Accidents in Past Year",
}


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_values = {}

    if not os.path.exists(MODEL_PATH):
        error = "Model not found. Please run 'python train_model.py' first to train and save the model."

    if request.method == "POST" and not error:
        try:
            segment = {
                "traffic_volume_per_day": float(request.form.get("traffic_volume_per_day", 0)),
                "speed_limit_kmh": float(request.form.get("speed_limit_kmh", 0)),
                "number_of_lanes": int(request.form.get("number_of_lanes", 1)),
                "has_streetlights": int(request.form.get("has_streetlights", 0)),
                "number_of_intersections": int(request.form.get("number_of_intersections", 0)),
                "pedestrian_crossings": int(request.form.get("pedestrian_crossings", 0)),
                "road_curvature_score": float(request.form.get("road_curvature_score", 0)),
                "road_condition_score": float(request.form.get("road_condition_score", 0)),
                "rainy_days_per_month": float(request.form.get("rainy_days_per_month", 0)),
                "historical_accidents_last_year": int(request.form.get("historical_accidents_last_year", 0)),
            }
            form_values = segment
            result = predict(segment)
        except (ValueError, TypeError):
            error = "Please make sure all fields contain valid numbers."

    importances = get_feature_importances() if not error else []

    return render_template(
        "index.html",
        result=result,
        error=error,
        form_values=form_values,
        importances=importances,
        field_labels=FIELD_LABELS,
    )


if __name__ == "__main__":
    app.run(debug=True, port=2000)
