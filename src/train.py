# src/train.py
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris

# Ensure data folder exists
os.makedirs("data/processed", exist_ok=True)

# Check if CSV exists; if not, create from sklearn dataset
csv_path = "data/processed/iris.csv"
if not os.path.isfile(csv_path):
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.to_csv(csv_path, index=False)
    print(f"Created dataset at {csv_path}")
else:
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset from {csv_path}")

# Prepare features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow experiment
mlflow.set_experiment("Iris-Classification")
with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {acc:.4f}")

    # Log metrics & model
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Save metrics locally (optional)
    os.makedirs("mlruns/metrics", exist_ok=True)
    with open("mlruns/metrics/metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f)

print("✅ Training complete and logged to MLflow.")

