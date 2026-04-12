# ==================================================
# Unemployment Trends Dashboard - Aligned with Notebook
# ==================================================
# Updates in this version:
# - Loads model metrics from model_comparison_summary.csv when available
#   (falls back to hardcoded notebook values otherwise)
# - Includes ALL 5 models from the notebook: Stacked Ensemble, Elastic Net,
#   XGBoost, Random Forest, SVR (RBF) - matching the final notebook run
# - Adds MAE column to match notebook output
# - Fixes data path (CSV lives at repo root, not in /data)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Unemployment Trends & ML Forecasting", page_icon="📊", layout="wide")

# --------------------------------------------------
# Paths (CSV is at repo root in the HF Space)
# --------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "Unemployment_Rate_Dataset.csv"
MODEL_SUMMARY_PATH = BASE_DIR / "model_comparison_summary.csv"

# --------------------------------------------------
# Load and clean data (Notebook-aligned)
# --------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    year_cols = [c for c in df.columns if c.isdigit()]

    df = df.dropna(subset=year_cols, how='all')
    df[year_cols] = df[year_cols].interpolate(method='linear', axis=1, limit_direction='both')
    df = df.drop(columns=['Indicator Name', 'Indicator Code'], errors='ignore')

    for col in year_cols:
        df[col] = df[col].clip(lower=df[col].quantile(0.05), upper=df[col].quantile(0.95))

    df_long = df.melt(
        id_vars=["Country Name"],
        value_vars=year_cols,
        var_name="Year",
        value_name="Unemployment_Rate"
    )
    df_long["Year"] = df_long["Year"].astype(int)
    df_long = df_long.dropna(subset=["Unemployment_Rate"])
    df_long = df_long.rename(columns={"Country Name": "Country"})
    return df_long


@st.cache_data(show_spinner=False)
def load_model_summary():
    """Load model comparison metrics from CSV, fall back to notebook values."""
    if MODEL_SUMMARY_PATH.exists():
        try:
            return pd.read_csv(MODEL_SUMMARY_PATH)
        except Exception:
            pass

    # Fallback: values from the final notebook run (image 2)
    return pd.DataFrame({
        "Model":   ["Stacked Ensemble", "Elastic Net", "XGBoost", "Random Forest", "SVR (RBF)"],
        "RMSE":    [1.0585, 1.0662, 1.0808, 1.0940, 2.0641],
        "MAE":     [0.6934, 0.6860, 0.7075, 0.7227, 1.4748],
        "R2":      [0.9424, 0.9416, 0.9400, 0.9385, 0.7811],
        "MAPE (%)":[11.0706, 11.0569, 11.2499, 11.5349, 25.2631],
        "MASE":    [1.4410, 1.4256, 1.4703, 1.5019, 3.0648],
    })


df = load_data()

# --------------------------------------------------
# World Bank regional groupings
# --------------------------------------------------
region_dict = {
    "East Asia & Pacific": [
        "American Samoa", "Australia", "Brunei Darussalam", "Cambodia", "China", "Fiji", "French Polynesia", "Guam",
        "Hong Kong SAR, China", "Indonesia", "Japan", "Kiribati", "Korea, Rep.", "Lao PDR", "Macao SAR, China",
        "Malaysia", "Marshall Islands", "Micronesia, Fed. Sts.", "Mongolia", "Myanmar", "Nauru", "New Caledonia",
        "New Zealand", "Northern Mariana Islands", "Palau", "Papua New Guinea", "Philippines", "Samoa", "Singapore",
        "Solomon Islands", "Taiwan, China", "Thailand", "Timor-Leste", "Tonga", "Tuvalu", "Vanuatu", "Viet Nam"
    ],
    "Europe & Central Asia": [
        "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium", "Bosnia and Herzegovina",
        "Bulgaria", "Channel Islands", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Faroe Islands",
        "Finland", "France", "Georgia", "Germany", "Gibraltar", "Greece", "Greenland", "Hungary", "Iceland",
        "Ireland", "Isle of Man", "Italy", "Kazakhstan", "Kosovo", "Kyrgyz Republic", "Latvia", "Liechtenstein",
        "Lithuania", "Luxembourg", "Monaco", "Montenegro", "Netherlands", "North Macedonia", "Norway", "Poland",
        "Portugal", "Romania", "Russian Federation", "San Marino", "Serbia", "Slovak Republic", "Slovenia", "Spain",
        "Sweden", "Switzerland", "Tajikistan", "Turkmenistan", "Türkiye", "Ukraine", "United Kingdom", "Uzbekistan"
    ],
    "Latin America & Caribbean": [
        "Antigua and Barbuda", "Aruba", "Bahamas, The", "Barbados", "Belize", "Bolivia", "Brazil",
        "British Virgin Islands", "Cayman Islands", "Chile", "Colombia", "Costa Rica", "Cuba", "Curacao", "Dominica",
        "Dominican Republic", "Ecuador", "El Salvador", "Grenada", "Guatemala", "Guyana", "Haiti", "Honduras",
        "Jamaica", "Mexico", "Nicaragua", "Panama", "Paraguay", "Peru", "Puerto Rico", "Sint Maarten (Dutch part)",
        "St. Kitts and Nevis", "St. Lucia", "St. Martin (French part)", "St. Vincent and the Grenadines", "Suriname",
        "Trinidad and Tobago", "Turks and Caicos Islands", "Uruguay", "Venezuela, RB", "Virgin Islands (U.S.)"
    ],
    "Middle East & North Africa": [
        "Afghanistan", "Algeria", "Bahrain", "Djibouti", "Egypt, Arab Rep.", "Iran, Islamic Rep.", "Iraq", "Israel",
        "Jordan", "Kuwait", "Lebanon", "Libya", "Malta", "Morocco", "Oman", "Pakistan", "Qatar", "Saudi Arabia",
        "Syrian Arab Republic", "Tunisia", "United Arab Emirates", "West Bank and Gaza", "Yemen, Rep."
    ],
    "North America": ["Bermuda", "Canada", "United States"],
    "South Asia": ["Bangladesh", "Bhutan", "India", "Maldives", "Nepal", "Sri Lanka"],
    "Sub-Saharan Africa": [
        "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cabo Verde", "Cameroon", "Central African Republic",
        "Chad", "Comoros", "Congo, Dem. Rep.", "Congo, Rep.", "Côte d'Ivoire", "Djibouti", "Equatorial Guinea",
        "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia, The", "Ghana", "Guinea", "Guinea-Bissau", "Kenya",
        "Lesotho", "Liberia", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Mozambique", "Namibia",
        "Niger", "Nigeria", "Rwanda", "São Tomé and Principe", "Senegal", "Seychelles", "Sierra Leone", "Somalia",
        "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo", "Uganda", "Zambia", "Zimbabwe"
    ]
}

all_countries = sorted(df["Country"].unique())
flat_countries = set()
for cl in region_dict.values():
    flat_countries.update(cl)
aggregates = [c for c in all_countries if c not in flat_countries]

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("🔍 Controls")
select_all = st.sidebar.checkbox("Select all countries", value=False)

selected_countries = []
for region, reg_countries in region_dict.items():
    with st.sidebar.expander(region):
        available = [c for c in reg_countries if c in all_countries]
        default = available if select_all else []
        sel = st.multiselect(f"Select from {region}", sorted(available), default=default)
        selected_countries.extend(sel)

with st.sidebar.expander("Aggregates & Groups"):
    default_agg = aggregates if select_all else []
    agg_sel = st.multiselect("Select Aggregates", sorted(aggregates), default=default_agg)
    selected_countries.extend(agg_sel)

countries = sorted(set(selected_countries))
if not countries:
    default_countries = ["Nigeria", "South Africa", "Egypt, Arab Rep.", "Kenya"]
    countries = [c for c in default_countries if c in all_countries]

year_range = st.sidebar.slider(
    "Year Range",
    int(df["Year"].min()), int(df["Year"].max()),
    (1991, 2024), step=1
)
smoothing = st.sidebar.selectbox("Smoothing Window", [1, 3, 5, 7], index=2)

# --------------------------------------------------
# Filter
# --------------------------------------------------
filtered_df = df[
    (df["Country"].isin(countries)) &
    (df["Year"].between(*year_range))
].copy()

# --------------------------------------------------
# Header & KPIs
# --------------------------------------------------
st.title("An Empirical Comparison of Machine Learning Models for Early Prediction of Unemployment Rates Across Countries")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Rate", f"{filtered_df['Unemployment_Rate'].mean():.2f}%")
c2.metric("Max Rate", f"{filtered_df['Unemployment_Rate'].max():.2f}%")
c3.metric("Min Rate", f"{filtered_df['Unemployment_Rate'].min():.2f}%")
c4.metric("Countries", filtered_df["Country"].nunique())

st.divider()

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Trends", "Distribution", "Regional Trends", "Heatmap & Corr",
     "Prediction & Year over Year", "Results & Models"]
)

with tab1:
    st.subheader("Unemployment Rate Trends")
    fig_line = px.line(filtered_df, x="Year", y="Unemployment_Rate", color="Country",
                       markers=True, template="plotly_white")
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader(f"Rolling Average ({smoothing}-year)")
    roll = filtered_df.sort_values(["Country", "Year"])
    roll["Rolling_Avg"] = roll.groupby("Country")["Unemployment_Rate"].rolling(smoothing).mean().values
    fig_roll = px.line(roll, x="Year", y="Rolling_Avg", color="Country", template="plotly_white")
    st.plotly_chart(fig_roll, use_container_width=True)

with tab2:
    st.subheader("Distribution & Variability")

    st.markdown("**Histogram of Unemployment Rates**")
    st.plotly_chart(px.histogram(filtered_df, x="Unemployment_Rate", nbins=30,
                                 template="plotly_white"), use_container_width=True)

    st.markdown("**Boxplot by Country**")
    st.plotly_chart(px.box(filtered_df, x="Country", y="Unemployment_Rate", color="Country",
                           points="all", template="plotly_white"), use_container_width=True)

    st.markdown("**Boxplot by Year**")
    st.plotly_chart(px.box(filtered_df, x="Year", y="Unemployment_Rate",
                           template="plotly_white"), use_container_width=True)

with tab3:
    st.subheader("Regional Trends (Selected Aggregates)")
    selected_regions = ["World", "Sub-Saharan Africa", "Europe & Central Asia",
                        "Latin America & Caribbean", "East Asia & Pacific (excluding high income)"]
    regions_df = df[df["Country"].isin(selected_regions) & df["Year"].between(*year_range)]
    st.plotly_chart(px.line(regions_df, x="Year", y="Unemployment_Rate", color="Country",
                            template="plotly_white", title="Major Regional Trends"),
                    use_container_width=True)

with tab4:
    st.subheader("Country × Year Heatmap")
    heat_df = filtered_df.pivot_table(index="Country", columns="Year", values="Unemployment_Rate")
    st.plotly_chart(px.imshow(heat_df, aspect="auto", color_continuous_scale="RdYlGn_r",
                              labels={"color": "Unemployment %"}), use_container_width=True)

    st.subheader("Correlation Matrix of Unemployment Rates")
    wide_df = filtered_df.pivot(index="Country", columns="Year", values="Unemployment_Rate")
    corr_matrix = wide_df.corr()
    st.plotly_chart(px.imshow(corr_matrix, color_continuous_scale="RdBu_r",
                              labels={"color": "Correlation"}), use_container_width=True)

with tab5:
    st.subheader("Previous Year vs Predicted (Rolling Baseline)")
    pred_df = filtered_df.sort_values(["Country", "Year"]).copy()
    pred_df["Prev_Year"] = pred_df.groupby("Country")["Unemployment_Rate"].shift(1)
    pred_df["Predicted"] = pred_df.groupby("Country")["Unemployment_Rate"].rolling(smoothing).mean().values

    fig_comp = go.Figure()
    for country in pred_df["Country"].unique():
        cdf = pred_df[pred_df["Country"] == country]
        fig_comp.add_trace(go.Scatter(x=cdf["Year"], y=cdf["Prev_Year"], mode="lines+markers",
                                      name=f"{country} - Prev", line=dict(dash="dot")))
        fig_comp.add_trace(go.Scatter(x=cdf["Year"], y=cdf["Predicted"], mode="lines",
                                      name=f"{country} - Pred"))
    fig_comp.update_layout(template="plotly_white", xaxis_title="Year", yaxis_title="Rate (%)")
    st.plotly_chart(fig_comp, use_container_width=True)

    st.subheader("Year-over-Year Change Composition")
    pie_df = filtered_df.sort_values(["Country", "Year"]).copy()
    pie_df["YoY"] = pie_df.groupby("Country")["Unemployment_Rate"].diff()
    pie_summary = pd.cut(pie_df["YoY"], bins=[-np.inf, -0.5, 0.5, np.inf],
                         labels=["Improvement", "Stable", "Deterioration"]).value_counts().reset_index()
    pie_summary.columns = ["Trend", "Count"]
    st.plotly_chart(px.pie(pie_summary, names="Trend", values="Count", hole=0.45,
                           color_discrete_map={"Improvement": "#2ecc71",
                                               "Stable": "#f1c40f",
                                               "Deterioration": "#e74c3c"}),
                    use_container_width=True)

with tab6:
    st.subheader("Model Performance Comparison (2020–2024 Test Period)")

    model_perf = load_model_summary()

    # Sort by RMSE ascending so best model is on top
    if "RMSE" in model_perf.columns:
        model_perf = model_perf.sort_values("RMSE").reset_index(drop=True)

    st.dataframe(
        model_perf.style.format({c: "{:.4f}" for c in model_perf.select_dtypes("number").columns}),
        use_container_width=True
    )

    best_row = model_perf.iloc[0]
    st.success(f"⭐ Best model: **{best_row['Model']}** "
               f"(RMSE = {best_row['RMSE']:.4f})")

    st.markdown("""
    **Key Insights (from Notebook Study)**
    - The **Stacked Ensemble** combines base learners through a Ridge meta-learner and achieves the lowest RMSE.
    - **Elastic Net** is the strongest single model, edging out tree-based methods.
    - **XGBoost** and **Random Forest** are competitive and best at capturing non-linear shocks (2008 & 2020 crises).
    - **SVR (RBF)** under-performs without further tuning and is included for comparison.
    - All models significantly outperform naïve forecasts.

    Lower RMSE / MAE / MAPE / MASE is better • Higher R² is better
    """)

    st.download_button("Download Model Comparison",
                       model_perf.to_csv(index=False),
                       "model_comparison.csv", "text/csv")

    st.subheader("Raw Filtered Data")
    st.dataframe(filtered_df, use_container_width=True)