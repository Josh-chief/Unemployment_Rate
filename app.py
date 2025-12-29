import gradio as gr
import pandas as pd
import joblib
import numpy as np

# =========================
# Load trained model
# =========================
model = joblib.load("xgb_unemployment_model.pkl")

# =========================
# Load datasets
# =========================
X = pd.read_csv("X_test_scaled_clean.csv")
raw_data = pd.read_csv("Unemployment_Rate_Dataset.csv")

COUNTRY_COLUMN = "Country Name"

# =========================
# Validate country column
# =========================
if COUNTRY_COLUMN not in raw_data.columns:
    raise ValueError(f"'{COUNTRY_COLUMN}' column not found in Unemployment_Rate_Dataset.csv")

# =========================
# Build country → row indices mapping
# =========================
country_to_indices = {}

for idx, country in raw_data[COUNTRY_COLUMN].items():
    country_to_indices.setdefault(country, []).append(idx)

countries = sorted(country_to_indices.keys())

# =========================
# Prediction function
# =========================
def predict_unemployment(country_name):
    indices = country_to_indices[country_name]

    # Select all rows for this country
    X_subset = X.iloc[indices]

    # Predict for all rows
    preds = model.predict(X_subset)

    # Aggregate prediction (mean)
    prediction_2025 = np.mean(preds)

    return f"{prediction_2025:.2f}%"

# =========================
# Gradio Interface
# =========================
app = gr.Interface(
    fn=predict_unemployment,
    inputs=gr.Dropdown(
        choices=countries,
        label="Select Country"
    ),
    outputs=gr.Textbox(
        label="Predicted Unemployment Rate for 2025",
        interactive=False
    ),
    title="Unemployment Rate Predictor (2025)",
    description=(
        "This application predicts the unemployment rate for 2025 using a trained "
        "XGBoost model. Predictions are aggregated across country-specific observations."
    )
)

if __name__ == "__main__":
    app.launch()
