# src/preprocess.py
import os
import pandas as pd
from sklearn.datasets import load_iris

# Ensure directories exist
os.makedirs("data/processed", exist_ok=True)

# Load dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Simple preprocessing (if needed)
# For Iris, we already have numeric features and target
df.to_csv("data/processed/iris.csv", index=False)
print("✅ Preprocessed dataset saved to data/processed/iris.csv")

