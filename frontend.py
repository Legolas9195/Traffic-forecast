import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from io import BytesIO

# -----------------------------
# APP UI
# -----------------------------
st.title("Traffic Forecast with Customer Adjustment")

# User inputs (with +/– clarification)
HIST_CUSTOMER_DROP = st.number_input(
    "Historical customer drop/increases (used for scaling) (+ for increase, – for drop)",
    value=425_000, step=10_000
)
FORECAST_CUSTOMER_DROP = st.number_input(
    "Expected customer drop/increases during forecast period (+ for increase, – for drop)",
    value=200_000, step=10_000
)
FORECAST_DAYS = st.number_input("Forecast horizon (days)", value=120, step=10)
SEASONAL_PERIODS = st.number_input("Seasonal periods (e.g., 30 for daily data)", value=30, step=1)

uploaded_file = st.file_uploader("Upload your CSV (with time_sec, AVG_Total_Traffic)", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    df["time_sec"] = pd.to_datetime(df["time_sec"])
    df.set_index("time_sec", inplace=True)
    series = df["AVG_Total_Traffic"].astype(float).sort_index()

    last_hist_date = series.index.max().normalize()

    # Fit Holt-Winters
    model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=SEASONAL_PERIODS)
    fit = model.fit()

    forecast = fit.forecast(FORECAST_DAYS)
    forecast_dates = pd.date_range(last_hist_date + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="D")

    forecast_df = pd.DataFrame({"time_sec": forecast_dates, "Forecast_raw": forecast.values})

    # Estimate traffic per customer (from historical change)
    start_mean = series.head(14).mean()
    end_mean = series.tail(14).mean()

    traffic_per_customer = 0.0
    if not np.isnan(start_mean) and not np.isnan(end_mean):
        change = start_mean - end_mean   # can be positive (drop) or negative (increase)
        est = change / HIST_CUSTOMER_DROP if HIST_CUSTOMER_DROP != 0 else 0
        traffic_per_customer = est

    # Apply customer adjustment linearly over forecast horizon
    forecast_df["Adjusted"] = forecast_df["Forecast_raw"]
    if FORECAST_CUSTOMER_DROP != 0 and traffic_per_customer != 0:
        daily_change = FORECAST_CUSTOMER_DROP / FORECAST_DAYS
        cum_change = np.arange(1, FORECAST_DAYS + 1) * daily_change
        adjustment = cum_change * traffic_per_customer
        forecast_df["Adjusted"] = forecast_df["Forecast_raw"].values - adjustment
        forecast_df["Adjusted"] = forecast_df["Adjusted"].clip(lower=0)

    # Merge with history
    hist_df = series.reset_index().rename(columns={"time_sec": "time_sec", "AVG_Total_Traffic": "Historical"})
    output_df = pd.concat([hist_df, forecast_df], ignore_index=True)

    # Show plots
    st.subheader("Forecast Chart")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(series.index, series.values, label="Historical")
    ax.plot(forecast_df["time_sec"], forecast_df["Forecast_raw"], "--", label="HW Forecast (raw)")
    ax.plot(
        forecast_df["time_sec"], forecast_df["Adjusted"],
        label="Adjusted Forecast (with customer change: + increase, – drop)"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Traffic")
    ax.legend()
    st.pyplot(fig)

    # Download Excel
    st.subheader("Download Results")
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        output_df.to_excel(writer, index=False, sheet_name="Forecast")
    st.download_button("Download Excel", data=output.getvalue(),
                       file_name="traffic_forecast_adjusted.xlsx", mime="application/vnd.ms-excel")

    # Info
    st.write(f"Estimated traffic change per customer: {traffic_per_customer:.6f}")
    st.write(f"Applied expected customer change: {FORECAST_CUSTOMER_DROP:,} customers (+ increase, – drop)")
