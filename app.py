import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Unemployment Trends Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# Load and clean data
# --------------------------------------------------
DATA_PATH = Path(__file__).parent / "data" / "Unemployment_Rate_Dataset.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # strip whitespace in headers
    df.columns = df.columns.str.strip()

    # --- detect country column automatically ---
    if "Country Name" in df.columns:
        country_col = "Country Name"
    elif "Country" in df.columns:
        country_col = "Country"
    elif "country" in df.columns:
        country_col = "country"
    else:
        st.error("No country column found in dataset.")
        st.stop()

    # --- detect year columns (numeric like 1990, 1991, etc.) ---
    year_cols = [c for c in df.columns if c.isdigit()]

    if len(year_cols) == 0:
        st.error("No numeric year columns found in dataset.")
        st.write("Columns detected:", df.columns.tolist())
        st.stop()

    # --- reshape: wide → long ---
    df_long = df.melt(
        id_vars=[country_col],
        value_vars=year_cols,
        var_name="Year",
        value_name="Unemployment_Rate"
    )

    df_long["Year"] = df_long["Year"].astype(int)
    df_long = df_long.dropna(subset=["Unemployment_Rate"])

    # rename country column consistently
    df_long = df_long.rename(columns={country_col: "Country"})

    return df_long

df = load_data()

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.title("🔍 Filters")

all_countries = sorted(df["Country"].unique())

default_countries = [
    c for c in all_countries
    if c in ["Nigeria", "South Africa", "Egypt, Arab Rep.", "Algeria"]
]

countries = st.sidebar.multiselect(
    "Select Countries",
    options=all_countries,
    default=default_countries
)

year_range = st.sidebar.slider(
    "Select Year Range",
    int(df["Year"].min()),
    int(df["Year"].max()),
    (2000, int(df["Year"].max()))
)

# filter data
filtered_df = df[
    (df["Country"].isin(countries)) &
    (df["Year"] >= year_range[0]) &
    (df["Year"] <= year_range[1])
]

# --------------------------------------------------
# Main title
# --------------------------------------------------
st.title("📊 Unemployment Rates Across African Economies")
st.markdown("Interactive analysis of unemployment trends — World Bank data")

# --------------------------------------------------
# KPI metrics
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Unemployment Rate",
    f"{filtered_df['Unemployment_Rate'].mean():.2f}%"
)

col2.metric(
    "Highest Recorded Rate",
    f"{filtered_df['Unemployment_Rate'].max():.2f}%"
)

col3.metric(
    "Lowest Recorded Rate",
    f"{filtered_df['Unemployment_Rate'].min():.2f}%"
)

st.divider()

# --------------------------------------------------
# Line chart
# --------------------------------------------------
st.subheader("📈 Unemployment Trends Over Time")

line_fig = px.line(
    filtered_df,
    x="Year",
    y="Unemployment_Rate",
    color="Country",
    markers=True,
    template="plotly_white"
)

st.plotly_chart(line_fig, use_container_width=True)

# --------------------------------------------------
# Bar chart – latest year
# --------------------------------------------------
st.subheader("📊 Country Comparison (Latest Year)")

latest_year = filtered_df["Year"].max()
latest_df = filtered_df[filtered_df["Year"] == latest_year]

bar_fig = px.bar(
    latest_df,
    x="Country",
    y="Unemployment_Rate",
    color="Country",
    text_auto=".2f",
    template="plotly_white"
)

st.plotly_chart(bar_fig, use_container_width=True)

# --------------------------------------------------
# Data table
# --------------------------------------------------
st.subheader("📄 Raw Data")
st.dataframe(filtered_df, use_container_width=True)
