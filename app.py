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
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from predictor import predict, get_feature_importances, MODEL_PATH

app = Flask(__name__)

# Database Configuration
db_host = os.environ.get("DB_HOST", "localhost")
db_port = os.environ.get("DB_PORT", "5432")
db_user = os.environ.get("DB_USER", "postgres")
db_password = os.environ.get("DB_PASSWORD", "postgres")
db_name = os.environ.get("DB_NAME", "accident_prediction_db")

app.config["SQLALCHEMY_DATABASE_URI"] = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

class PredictionHistory(db.Model):
    __tablename__ = "prediction_history"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Inputs
    traffic_volume_per_day = db.Column(db.Float, nullable=False)
    speed_limit_kmh = db.Column(db.Float, nullable=False)
    number_of_lanes = db.Column(db.Integer, nullable=False)
    has_streetlights = db.Column(db.Integer, nullable=False)
    number_of_intersections = db.Column(db.Integer, nullable=False)
    pedestrian_crossings = db.Column(db.Integer, nullable=False)
    road_curvature_score = db.Column(db.Float, nullable=False)
    road_condition_score = db.Column(db.Float, nullable=False)
    rainy_days_per_month = db.Column(db.Float, nullable=False)
    historical_accidents_last_year = db.Column(db.Integer, nullable=False)
    
    # Outputs
    risk_probability = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(10), nullable=False)
    risk_factors = db.Column(db.JSON, nullable=False)

# Initialize database tables
with app.app_context():
    try:
        db.create_all()
        print("Database tables initialized successfully.")
    except Exception as e:
        print(f"Warning: Could not connect to PostgreSQL or initialize database tables: {e}")
        print("Ensure PostgreSQL is running and the database 'accident_prediction_db' has been created.")

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
            
            # Save prediction to PostgreSQL history log
            try:
                prediction_record = PredictionHistory(
                    traffic_volume_per_day=segment["traffic_volume_per_day"],
                    speed_limit_kmh=segment["speed_limit_kmh"],
                    number_of_lanes=segment["number_of_lanes"],
                    has_streetlights=segment["has_streetlights"],
                    number_of_intersections=segment["number_of_intersections"],
                    pedestrian_crossings=segment["pedestrian_crossings"],
                    road_curvature_score=segment["road_curvature_score"],
                    road_condition_score=segment["road_condition_score"],
                    rainy_days_per_month=segment["rainy_days_per_month"],
                    historical_accidents_last_year=segment["historical_accidents_last_year"],
                    risk_probability=result["risk_probability"],
                    risk_level=result["risk_level"],
                    risk_factors=result["risk_factors"]
                )
                db.session.add(prediction_record)
                db.session.commit()
            except Exception as db_err:
                print(f"Warning: Failed to log prediction to PostgreSQL database: {db_err}")
                
        except (ValueError, TypeError):
            error = "Please make sure all fields contain valid numbers."

    importances = get_feature_importances() if not error else []

    recent_predictions = []
    try:
        recent_predictions = PredictionHistory.query.order_by(PredictionHistory.timestamp.desc()).limit(10).all()
    except Exception as db_err:
        print(f"Warning: Failed to fetch prediction history: {db_err}")

    return render_template(
        "index.html",
        result=result,
        error=error,
        form_values=form_values,
        importances=importances,
        field_labels=FIELD_LABELS,
        history=recent_predictions,
    )


if __name__ == "__main__":
    app.run(debug=True, port=2000)

