import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1250px;}
.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #0f2b46 0%, #1f4e79 100%);
    color: white;
    margin-bottom: 1.2rem;
}
.hero h1 {margin: 0; font-size: 2.4rem;}
.hero p {margin: .35rem 0 0; font-size: 1.05rem; opacity: .92;}
.card {
    padding: 1rem 1.1rem;
    border: 1px solid rgba(100,100,100,.18);
    border-radius: 14px;
    background: rgba(255,255,255,.03);
}
.metric-title {font-size: .9rem; opacity: .72;}
.metric-value {font-size: 1.45rem; font-weight: 700;}
.small-note {font-size: .88rem; opacity: .72;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "random_forest_model.pkl")
    imputer = joblib.load(MODEL_DIR / "imputer.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    meta_path = MODEL_DIR / "metadata.json"
    metrics_path = MODEL_DIR / "metrics.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return model, imputer, list(feature_names), metadata, metrics

try:
    model, imputer, feature_names, metadata, metrics = load_artifacts()
except Exception as exc:
    st.error("Model files are missing or could not be loaded.")
    st.code(str(exc))
    st.stop()

has_model_feature = any(str(c).startswith("Model_") for c in feature_names)

# ---------- Sidebar ----------
st.sidebar.title("🚗 Car Predictor")
page = st.sidebar.radio(
    "Project pages",
    ["🏠 Home", "🔮 Predict Price", "📊 Model Performance", "🧠 How It Works", "📁 Dataset & Project"],
)


# ---------- Home ----------
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
      <h1>🚗 Used Car Price Predictor</h1>
      <p>Estimate the market price of a used car using a trained Random Forest regression model.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("🤖 Model", "Random Forest"),
        ("📊 Task", "Regression"),
        ("💰 Output", "Price in ₹ Lakhs"),
        ("🧩 Model input", "Included" if has_model_feature else "Needs retraining"),
    ]
    for col, (title, value) in zip([c1,c2,c3,c4], cards):
        with col:
            st.markdown(f'<div class="card"><div class="metric-title">{title}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

    st.subheader("What this project does")
    st.write(
        "Enter the vehicle's specifications, validate the inputs, convert categorical values "
        "into the same encoded feature structure used during training, and generate an estimated price."
    )

    if not has_model_feature:
        st.warning(
            "The currently loaded model was trained without the new Model feature. "
            "Run train_model.py once with your original dataset to activate Model selection."
        )

    st.subheader("Project highlights")
    a, b, c = st.columns(3)
    with a:
        st.markdown("### 🧹 Data preparation")
        st.write("Missing-value handling, numeric conversion, feature engineering and encoding.")
    with b:
        st.markdown("### 🌲 Random Forest")
        st.write("An ensemble of decision trees learns non-linear price relationships.")
    with c:
        st.markdown("### 🖥️ Streamlit UI")
        st.write("A simple multi-page interface designed for project demonstration and viva.")

# ---------- Prediction ----------
elif page == "🔮 Predict Price":
    st.markdown("## 🔮 Estimate Used-Car Price")
    st.caption("All inputs are validated before prediction.")

    # Metadata created by train_model.py. Fallbacks keep the old project usable.
    brands = metadata.get("brands", [])
    models_by_brand = metadata.get("models_by_brand", {})
    models = metadata.get("models", [])
    locations = metadata.get("locations", [])
    fuels = metadata.get("fuel_types", [])
    transmissions = metadata.get("transmissions", [])
    owners = metadata.get("owner_types", [])

    if not brands:
        brands = ["Audi", "BMW", "Ford", "Honda", "Hyundai", "Mahindra", "Maruti", "Mercedes-Benz", "Nissan", "Renault", "Skoda", "Tata", "Toyota", "Volkswagen"]
    if not locations:
        locations = ["Bangalore", "Chennai", "Coimbatore", "Delhi", "Hyderabad", "Jaipur", "Kochi", "Kolkata", "Mumbai", "Pune"]
    if not fuels:
        fuels = ["Diesel", "Petrol", "CNG", "LPG", "Electric"]
    if not transmissions:
        transmissions = ["Automatic", "Manual"]
    if not owners:
        owners = ["First", "Second", "Third", "Fourth & Above"]

    # Keep the Brand and Model selectors outside the form so Streamlit reruns
    # immediately when Brand changes. This makes the Model list truly
    # dependent on the selected Brand instead of retaining a stale value.
    st.subheader("Vehicle details")
    left, right = st.columns(2)

    with left:
        year = st.number_input("Manufacturing Year", min_value=1990, max_value=datetime.now().year, value=2015, step=1)
        kilometers = st.number_input("Kilometers Driven", min_value=0, max_value=1_000_000, value=50_000, step=1000)
        mileage = st.number_input("Mileage (km/l or km/kg)", min_value=1.0, max_value=60.0, value=15.0, step=0.1)
        engine = st.number_input("Engine Capacity (CC)", min_value=500.0, max_value=8000.0, value=1200.0, step=50.0)
        power = st.number_input("Power (bhp)", min_value=20.0, max_value=1000.0, value=80.0, step=1.0)
        seats = st.number_input("Number of Seats", min_value=2, max_value=10, value=5, step=1)

    with right:
        brand = st.selectbox("Car Brand", brands, key="brand_select")
        brand_models = models_by_brand.get(brand, [])
        if not brand_models:
            brand_models = models

        # The model options are rebuilt from the current Brand on every
        # Streamlit rerun. A deterministic key per brand avoids stale choices.
        model_key = f"model_select_{brand}"
        if has_model_feature and brand_models:
            model_name = st.selectbox("Car Model", brand_models, key=model_key)
        else:
            model_name = None

        location = st.selectbox("Location", locations)
        fuel = st.selectbox("Fuel Type", fuels)
        transmission = st.selectbox("Transmission", transmissions)
        owner = st.selectbox("Owner Type", owners)

    submitted = st.button("🔮 Predict Price", use_container_width=True)

    if submitted:
        errors = []
        current_year = datetime.now().year

        if not (1990 <= year <= current_year):
            errors.append("Manufacturing year is outside the valid range.")
        if kilometers < 0:
            errors.append("Kilometers driven cannot be negative.")
        if not (1 <= mileage <= 60):
            errors.append("Mileage should be between 1 and 60.")
        if not (500 <= engine <= 8000):
            errors.append("Engine capacity should be between 500 and 8000 CC.")
        if not (20 <= power <= 1000):
            errors.append("Power should be between 20 and 1000 bhp.")
        if not (2 <= seats <= 10):
            errors.append("Seats should be between 2 and 10.")

        if errors:
            for err in errors:
                st.error(err)
            st.stop()

        input_data = pd.DataFrame(0.0, index=[0], columns=feature_names)

        numeric_values = {
            "Year": year,
            "Kilometers_Driven": kilometers,
            "Mileage": mileage,
            "Engine": engine,
            "Power": power,
            "Seats": seats,
            "Car_Age": current_year - year,
        }
        for col, val in numeric_values.items():
            if col in input_data.columns:
                input_data[col] = val

        def set_dummy(prefix, value):
            col = f"{prefix}_{value}"
            if col in input_data.columns:
                input_data[col] = 1.0

        set_dummy("Brand", brand)
        set_dummy("Location", location)
        set_dummy("Fuel_Type", fuel)
        if transmission == "Manual" and "Transmission_Manual" in input_data.columns:
            input_data["Transmission_Manual"] = 1.0
        if owner != "First":
            set_dummy("Owner_Type", owner)

        if has_model_feature and model_name:
            set_dummy("Model", model_name)

        try:
            prepared = pd.DataFrame(imputer.transform(input_data), columns=feature_names)
            prediction = float(model.predict(prepared)[0])
            if not np.isfinite(prediction):
                raise ValueError("Model returned an invalid number.")

            st.success(f"### Estimated Used-Car Price: ₹ {prediction:.2f} Lakhs")
            st.caption("This is a model estimate, not a guaranteed selling price.")

            s1, s2, s3 = st.columns(3)
            s1.metric("Car Age", f"{current_year - year} years")
            s2.metric("Distance", f"{kilometers:,.0f} km")
            s3.metric("Selected Model", model_name if model_name else "Legacy model")
        except Exception as exc:
            st.error("Prediction failed. Please check that the saved model artifacts match the current application.")
            st.code(str(exc))

# ---------- Model performance ----------
elif page == "📊 Model Performance":
    st.markdown("## 📊 Model Performance")

    if metrics:
        rf = metrics.get("random_forest", {})
        baseline = metrics.get("linear_regression_baseline", {})

        c1, c2, c3 = st.columns(3)
        c1.metric("Random Forest MAE", f"{rf.get('MAE', '—'):.3f}" if isinstance(rf.get("MAE"), (int,float)) else "—")
        c2.metric("Random Forest RMSE", f"{rf.get('RMSE', '—'):.3f}" if isinstance(rf.get("RMSE"), (int,float)) else "—")
        c3.metric("Random Forest R²", f"{rf.get('R2', '—'):.3f}" if isinstance(rf.get("R2"), (int,float)) else "—")

        st.subheader("Evaluation summary")
        rows = []
        if baseline:
            rows.append({
                "Model": "Linear Regression",
                "Role": "Baseline",
                "MAE": baseline.get("MAE"),
                "RMSE": baseline.get("RMSE"),
                "R²": baseline.get("R2"),
            })
        rows.append({
            "Model": "Random Forest",
            "Role": "Final model",
            "MAE": rf.get("MAE"),
            "RMSE": rf.get("RMSE"),
            "R²": rf.get("R2"),
        })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.info("Lower MAE/RMSE is better. Higher R² is better.")
    else:
        st.warning("No verified final-model metrics are available yet.")
        st.write("Run `train_model.py` with the original training dataset. It will generate `model/metrics.json` automatically.")
        st.subheader("Known baseline from the earlier project")
        st.metric("Linear Regression R²", "0.569")
        st.write("Recorded baseline: MAE ≈ 3.145, RMSE ≈ 7.28, R² ≈ 0.569.")

# ---------- How it works ----------
elif page == "🧠 How It Works":
    st.markdown("## 🧠 How the Predictor Works")
    steps = [
        ("1", "User input", "Vehicle details are collected through the Streamlit form."),
        ("2", "Validation", "Ranges and required selections are checked before prediction."),
        ("3", "Feature engineering", "Car age is calculated and categorical values are mapped to model columns."),
        ("4", "Imputation", "The saved imputer applies the same missing-value treatment used during training."),
        ("5", "Random Forest", "The trained ensemble predicts the used-car price."),
        ("6", "Result", "The estimate is displayed in Indian lakhs."),
    ]
    for n, title, desc in steps:
        st.markdown(f"### {n}. {title}")
        st.write(desc)

    st.subheader("Why Model matters")
    st.write(
        "The original dataset contains a `Name` field that combines the car brand and model. "
        "The improved training pipeline splits it into `Brand` and `Model`, allowing the exact model "
        "selection to contribute to the prediction instead of using brand alone."
    )

# ---------- Dataset ----------
else:
    st.markdown("## 📁 Dataset & Project")
    st.write(
        "The project uses the Indian used-car dataset with vehicle attributes such as Name, Location, "
        "Year, Kilometers_Driven, Fuel_Type, Transmission, Owner_Type, Mileage, Engine, Power, Seats, "
        "New_Price and Price."
    )

    st.subheader("Current feature status")
    feature_status = pd.DataFrame([
        ["Brand", "Yes", "Extracted from Name"],
        ["Model", "Yes" if has_model_feature else "Not in current saved model", "Requires retraining" if not has_model_feature else "Encoded and trained"],
        ["Car Age", "Yes", "Derived from manufacturing year"],
        ["Input validation", "Yes", "Implemented in UI"],
        ["Random Forest metrics", "Yes" if metrics else "Pending", "Read from metrics.json"],
    ], columns=["Feature", "Status", "Implementation"])
    st.dataframe(feature_status, use_container_width=True, hide_index=True)

    st.subheader("Saved artifacts")
    for name in ["random_forest_model.pkl", "imputer.pkl", "feature_names.pkl", "metadata.json", "metrics.json"]:
        path = MODEL_DIR / name
        st.write(("✅ " if path.exists() else "⬜ ") + name)
