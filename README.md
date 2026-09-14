# Used Car Price Predictor — Model-Enhanced Final Pack

## What is now implemented

### 1. Car Model is a real ML feature
The original `Name` column is split into:
- **Brand** — first token of `Name`
- **Model** — remaining text of `Name`

`Model` is one-hot encoded and included in the Random Forest training features. The Streamlit app also shows a **Car Model** dropdown filtered by the selected brand, so the selected model actually changes the encoded input.

### 2. Actual dataset used
- Training rows loaded: **6,019**
- Columns: **13**
- Rows remaining after project cleaning/validation: **5,949**
- Model-ready features: **1,895**

### 3. Verified test performance
Using an 80/20 train-test split with `random_state=42`:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 3.687 | 7.514 | 0.514 |
| Random Forest | **1.434** | **3.135** | **0.915** |

The Random Forest is the final model because it gives lower error and higher R² on the held-out test set.

### 4. Input validation
The app validates year, kilometres, mileage, engine capacity, power and seats before prediction.

### 5. Streamlit UI
Pages included:
- Home
- Predict Price
- Model Performance
- How It Works
- Dataset & Project

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The trained artifacts are already included in the `model/` folder.

## Retraining later

If you receive a new version of the training dataset:

```bash
python train_model.py --data Data_Train.xlsx
```

This regenerates the model, imputer, feature names, metrics, metadata and feature-importance files.

## Important project note

The model now genuinely uses the **Model** feature. Do not add a Model dropdown without retraining, because a UI-only field would not affect predictions.
