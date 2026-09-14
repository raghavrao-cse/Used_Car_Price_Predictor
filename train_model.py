"""
Retrain the Used Car Price Predictor with the Model feature.

Usage:
    python train_model.py --data Data_Train.xlsx

The script:
1. Loads the original training dataset.
2. Confirms that Name exists.
3. Splits Name into Brand + Model.
4. Cleans numeric fields.
5. Creates Car_Age.
6. One-hot encodes categorical fields, including Model.
7. Trains Linear Regression and Random Forest.
8. Saves model artifacts, metrics and category metadata for the Streamlit app.
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)

def clean_number(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else np.nan

def prepare_data(df):
    required = [
        "Name", "Location", "Year", "Kilometers_Driven", "Fuel_Type",
        "Transmission", "Owner_Type", "Mileage", "Engine", "Power", "Seats", "Price"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data = df.copy()
    data["Name"] = data["Name"].fillna("Unknown Unknown").astype(str).str.strip()

    # Same basic convention used by the original project:
    # first token = Brand, remaining text = Model.
    parts = data["Name"].str.split(n=1, expand=True)
    data["Brand"] = parts[0].fillna("Unknown")
    data["Model"] = parts[1].fillna("Unknown").str.strip()

    for col in ["Mileage", "Engine", "Power", "Year", "Kilometers_Driven", "Seats", "Price"]:
        data[col] = data[col].apply(clean_number)

    data["Car_Age"] = pd.Timestamp.now().year - data["Year"]

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["Price"])

    # Remove impossible training values.
    data = data[(data["Price"] > 0)]
    data = data[(data["Year"] >= 1990) & (data["Year"] <= pd.Timestamp.now().year)]
    data = data[(data["Kilometers_Driven"] >= 0)]
    data = data[(data["Mileage"].isna()) | ((data["Mileage"] > 0) & (data["Mileage"] <= 60))]
    data = data[(data["Engine"].isna()) | ((data["Engine"] >= 500) & (data["Engine"] <= 8000))]
    data = data[(data["Power"].isna()) | ((data["Power"] >= 20) & (data["Power"] <= 1000))]
    data = data[(data["Seats"].isna()) | ((data["Seats"] >= 2) & (data["Seats"] <= 10))]

    numeric = ["Year", "Kilometers_Driven", "Mileage", "Engine", "Power", "Seats", "Car_Age"]
    categorical = ["Brand", "Model", "Location", "Fuel_Type", "Transmission", "Owner_Type"]

    X = data[numeric + categorical].copy()
    y = data["Price"].astype(float)

    X = pd.get_dummies(X, columns=categorical, drop_first=True, dtype=float)

    # Keep deterministic feature order.
    X = X.reindex(sorted(X.columns), axis=1)

    return X, y, data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Data_Train.xlsx or the training CSV")
    args = parser.parse_args()

    data_path = Path(args.data)
    if data_path.suffix.lower() in [".xlsx", ".xls"]:
        raw = pd.read_excel(data_path)
    else:
        raw = pd.read_csv(data_path)

    if "Name" not in raw.columns:
        raise ValueError("The supplied dataset does not contain a Name column, so Model cannot be added.")

    print(f"Loaded dataset: {raw.shape[0]} rows × {raw.shape[1]} columns")
    print("Name column found: YES")
    print("Model will be derived from Name: YES")

    X, y, cleaned = prepare_data(raw)
    print(f"Rows after cleaning: {len(cleaned)}")
    print(f"Model-ready features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Baseline
    lr = LinearRegression()
    lr.fit(X_train_imp, y_train)
    lr_pred = lr.predict(X_test_imp)

    # Final model
    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1,
        max_features="sqrt",
    )
    rf.fit(X_train_imp, y_train)
    rf_pred = rf.predict(X_test_imp)

    def metric_dict(y_true, pred):
        return {
            "MAE": float(mean_absolute_error(y_true, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
            "R2": float(r2_score(y_true, pred)),
        }

    metrics = {
        "random_forest": metric_dict(y_test, rf_pred),
        "linear_regression_baseline": metric_dict(y_test, lr_pred),
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
        "feature_count": int(X.shape[1]),
        "target_unit": "Indian Lakhs",
        "random_state": 42,
    }

    feature_names = list(X.columns)
    joblib.dump(rf, MODEL_DIR / "random_forest_model.pkl")
    joblib.dump(imputer, MODEL_DIR / "imputer.pkl")
    joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")
    joblib.dump(lr, MODEL_DIR / "linear_regression_model.pkl")

    metadata = {
        "brands": sorted(cleaned["Brand"].dropna().unique().tolist()),
        "models": sorted(cleaned["Model"].dropna().unique().tolist()),
        "locations": sorted(cleaned["Location"].dropna().unique().tolist()),
        "fuel_types": sorted(cleaned["Fuel_Type"].dropna().unique().tolist()),
        "transmissions": sorted(cleaned["Transmission"].dropna().unique().tolist()),
        "owner_types": sorted(cleaned["Owner_Type"].dropna().unique().tolist()),
        "has_model_feature": bool(any(str(c).startswith("Model_") for c in feature_names)),
        "dataset_rows_after_cleaning": int(len(cleaned)),
        "trained_at": pd.Timestamp.now().isoformat(),
    }

    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Feature importance for the UI/report.
    importance = (
        pd.DataFrame({"feature": feature_names, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    print("\n=== VERIFIED TEST METRICS ===")
    print("Linear Regression:", metrics["linear_regression_baseline"])
    print("Random Forest:", metrics["random_forest"])
    print("\nSaved:")
    for name in [
        "random_forest_model.pkl",
        "linear_regression_model.pkl",
        "imputer.pkl",
        "feature_names.pkl",
        "metrics.json",
        "metadata.json",
        "feature_importance.csv",
    ]:
        print(" -", MODEL_DIR / name)

if __name__ == "__main__":
    main()
