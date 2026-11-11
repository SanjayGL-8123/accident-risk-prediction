# 🚦 Accident Risk Prediction - Project & Remaining Tasks

This repository contains a Machine Learning-powered system to estimate traffic accident risks on various road segments based on environmental factors, historical data, and traffic volume.

> [!NOTE]
> Currently, the project utilizes a synthetic dataset for demonstration and pipeline building. To deploy in a production scenario, you should replace this with real-world traffic data.

---

## 🛠️ Project Structure & Architecture

The project consists of the following components:

*   **[dataset.py](file:///c:/Users/Sanjay G L/Desktop/accident-risk-prediction/dataset.py)**: Generates synthetic road-segment data containing 10 safety/traffic features and a `high_risk` classification label using a weighted risk formula.
*   **[train_model.py](file:///c:/Users/Sanjay G L/Desktop/accident-risk-prediction/train_model.py)**: Trains a Random Forest Classifier using `scikit-learn`, prints metrics (Accuracy, ROC-AUC, classification report), and dumps the serialized model package to `model/accident_risk_model.joblib`.
*   **[predictor.py](file:///c:/Users/Sanjay G L/Desktop/accident-risk-prediction/predictor.py)**: Loads the trained model, performs predictions, converts probability scores to qualitative risk levels (Low/Medium/High), and identifies rule-based risk factors to explain predictions.
*   **[app.py](file:///c:/Users/Sanjay G L/Desktop/accident-risk-prediction/app.py)**: A lightweight Flask web application that serves a form where users can input road features to see a visual risk prediction gauge, list of risk flags, and feature importances.
*   **[templates/index.html](file:///c:/Users/Sanjay G L/Desktop/accident-risk-prediction/templates/index.html)**: The frontend user interface for rendering inputs, prediction results, progress trackers, and model weights.

---

## 🚀 Setup & Execution

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Train and save the classifier (required before running the web app):
```bash
python train_model.py
```

### 3. Run the Web Server
Start the Flask application:
```bash
python app.py
```
Open **[http://127.0.0.1:2000](http://127.0.0.1:2000)** in your browser to interact with the interface.

### 4. Database Setup (PostgreSQL)
By default, the application runs on a local PostgreSQL connection with credentials `postgres/postgres` and database name `accident_prediction_db`.

Make sure to create the database in PostgreSQL before running the server:
```sql
CREATE DATABASE accident_prediction_db;
```

To configure custom settings, set the following environment variables in your terminal:
*   `DB_HOST` (default: `localhost`)
*   `DB_PORT` (default: `5432`)
*   `DB_USER` (default: `postgres`)
*   `DB_PASSWORD` (default: `postgres`)
*   `DB_NAME` (default: `accident_prediction_db`)

---

## 📋 Remaining Tasks (To-Do List)

Below is the roadmap of remaining enhancements to make the Accident Risk Prediction system more robust, feature-complete, and production-ready:

### Phase 1: Machine Learning & Dataset Enhancements
- [ ] **Hyperparameter Tuning**: Implement grid or randomized search (`GridSearchCV` / `RandomizedSearchCV`) in [train_model.py](file:///c:/Users/Sanjay G L/Desktop/accident-risk-prediction/train_model.py) to optimize Random Forest parameters (like `n_estimators` and `max_depth`).
- [ ] **Cross-Validation**: Integrate k-fold cross-validation during training to ensure model stability and avoid overfitting.
- [ ] **Real Data Ingestion**: Modify [dataset.py](file:///c:/Users/Sanjay G L/Desktop/accident-risk-prediction/dataset.py) to ingest real-world road traffic records from CSV or a database rather than generating synthetic data.
- [ ] **Model Comparison**: Experiment with and evaluate alternative models such as Gradient Boosting (XGBoost, LightGBM) or Logistic Regression.

### Phase 2: Frontend & UI Improvements
- [ ] **Dynamic Visualization**: Replace the CSS progress-bar feature importances in [templates/index.html](file:///c:/Users/Sanjay G L/Desktop/accident-risk-prediction/templates/index.html) with dynamic charts using a library like Chart.js or Plotly.js.
- [ ] **Form Validations**: Add strict HTML5/JavaScript boundaries to ensure users can only input realistic values (e.g. speed limits between 10 and 150 km/h, road score between 0 and 10).
- [ ] **Responsive Refinements**: Polish the page layout to render perfectly on small mobile screens.

### Phase 3: Backend API & Deployment
- [ ] **API Endpoint**: Expose a REST API endpoint (`POST /api/predict`) in [app.py](file:///c:/Users/Sanjay G L/Desktop/accident-risk-prediction/app.py) returning prediction payloads in JSON format.
- [x] **Prediction History Logging**: Integrated PostgreSQL database logging with Flask-SQLAlchemy to record prediction histories.
-
- [ ] **Unit Testing**: Create a suite of automated unit tests using `pytest` to test the dataset generation, training sequence, and predictor utility logic.
