"""
train_model.py
---------------
Trains a Random Forest classifier on the synthetic road segment dataset and
saves it to disk with joblib, along with feature importances for the UI.

Run with:
    python train_model.py
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from dataset import generate_dataset, to_feature_matrix, FEATURE_NAMES


def main():
    data = generate_dataset(n_samples=800, random_state=42)
    X, y = to_feature_matrix(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=7,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print(f"Test accuracy: {accuracy:.2%}")
    print(f"ROC-AUC: {auc:.3f}")
    print(classification_report(y_test, predictions, target_names=["LOW RISK", "HIGH RISK"]))

    importances = dict(zip(FEATURE_NAMES, model.feature_importances_))
    print("\nFeature importances:")
    for name, importance in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {name}: {importance:.3f}")

    joblib.dump({
        "model": model,
        "feature_names": FEATURE_NAMES,
        "feature_importances": importances,
    }, "model/accident_risk_model.joblib")

    print("\nModel saved to model/accident_risk_model.joblib")


if __name__ == "__main__":
    main()
