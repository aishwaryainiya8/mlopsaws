# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc

app = FastAPI(title="Iris Classifier API")

# =====================================================
# 1️⃣ Define schema for a single sample
# =====================================================
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm", example=5.1)
    sepal_width: float = Field(..., description="Sepal width in cm", example=3.5)
    petal_length: float = Field(..., description="Petal length in cm", example=1.4)
    petal_width: float = Field(..., description="Petal width in cm", example=0.2)

    class Config:
        extra = "forbid"  # Disallow extra keys


# =====================================================
# 2️⃣ Request schema for batch predictions
# =====================================================
class PredictRequest(BaseModel):
    data: list[IrisFeatures] = Field(
        ...,
        example=[
            {
                "sepal_length": 6.2,
                "sepal_width": 3.4,
                "petal_length": 5.4,
                "petal_width": 2.3
            }
        ],
        description="List of Iris feature data",
    )

    @model_validator(mode="after")
    def ensure_not_empty(self):
        if not self.data:
            raise ValueError("data cannot be empty")
        return self


# =====================================================
# 3️⃣ Load model from MLflow
# =====================================================
CLASS_NAMES = ["setosa", "versicolor", "virginica"]
experiment_name = "Iris-Classification"
mlflow.set_experiment(experiment_name)

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
if not experiment:
    raise Exception(f"No MLflow experiment found with name '{experiment_name}'")

runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
if not runs:
    raise Exception(f"No MLflow runs found for experiment '{experiment_name}'")

latest_run_id = runs[0].info.run_id
model_uri = f"runs:/{latest_run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)
print(f"✅ Loaded model from run {latest_run_id}")


# =====================================================
# 4️⃣ API endpoints
# =====================================================
@app.get("/")
def read_root():
    return {"status": "API running", "experiment": experiment_name, "run_id": latest_run_id}


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Accepts a JSON list of feature dictionaries, predicts class and confidence.
    Example input:
    {
      "data": [
        {"sepal_length": 6.2, "sepal_width": 3.4, "petal_length": 5.4, "petal_width": 2.3}
      ]
    }
    """
    try:
        X = pd.DataFrame([r.model_dump() for r in request.data])

        rename_map = {
            "sepal_length": "sepal length (cm)",
            "sepal_width": "sepal width (cm)",
            "petal_length": "petal length (cm)",
            "petal_width": "petal width (cm)"
        }
        X = X.rename(columns=rename_map)

        preds = model.predict(X)

        confidences = [None] * len(preds)
        try:
            sk_model = model._model_impl
            if hasattr(sk_model, "predict_proba"):
                probas = sk_model.predict_proba(X)
                confidences = np.max(probas, axis=1)
        except Exception:
            pass

        results = [
            {
                "predicted_class": CLASS_NAMES[int(pred)],
                "confidence": float(conf) if conf is not None else None
            }
            for pred, conf in zip(preds, confidences)
        ]

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

