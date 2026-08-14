import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import timedelta
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Air Quality Prediction", layout="wide")

st.title("🌍 Air Quality Prediction System")
st.markdown("👤 **Author:** Nurikamal Bolatbay")
st.markdown("---")

# ============================================================
# HELPER FUNCTIONS FOR READING CSV
# ============================================================

def find_column(df, possible_names):
    """Find column by exact or partial match."""
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    for name in possible_names:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    for c in df.columns:
        c_lower = str(c).strip().lower()
        for name in possible_names:
            if name.lower() in c_lower:
                return c
    return None

def read_csv_safe(filename):
    """Try to read CSV with multiple separators and encodings."""
    if not os.path.exists(filename):
        return None
    for sep in [',', ';', '\t']:
        for enc in ['utf-8', 'latin1', 'cp1252']:
            try:
                df = pd.read_csv(filename, sep=sep, encoding=enc, on_bad_lines='skip', low_memory=False)
                if df.shape[1] >= 2:
                    return df
            except:
                pass
    return None

def standardize_dataset(df, forced_city=None):
    """
    Convert any dataset to a standard format: time, city, pm25.
    Returns None if conversion fails.
    """
    if df is None or df.empty:
        return None
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    
    # Time column
    time_col = find_column(df, ["time", "datetime", "date", "timestamp", "last_updated"])
    if time_col is None:
        # Try first column
        possible = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        if possible.notna().sum() > len(df) * 0.5:
            time_col = df.columns[0]
    if time_col is None:
        return None
    
    df["time"] = pd.to_datetime(df[time_col], errors='coerce', utc=True)
    try:
        df["time"] = df["time"].dt.tz_localize(None)
    except:
        pass
    
    # PM2.5 column
    pm_col = find_column(df, ["pm2.5", "pm25", "pm2_5", "pm_2_5", "pm 2.5", "pm2"])
    if pm_col is None:
        return None
    df["pm25"] = pd.to_numeric(df[pm_col], errors='coerce')
    
    # City column
    city_col = find_column(df, ["city", "location", "place", "site"])
    if forced_city is not None:
        df["city"] = forced_city
    elif city_col is not None:
        df["city"] = df[city_col].astype(str).str.strip()
    else:
        df["city"] = "Unknown"
    
    df = df[["time", "city", "pm25"]].copy()
    df = df.dropna(subset=["time", "pm25"])
    df = df[(df["pm25"] >= 0) & (df["pm25"] <= 1000)]
    df = df.sort_values("time")
    if df.empty:
        return None
    return df

# ============================================================
# LOAD DATA FROM FILES (NO SYNTHETIC)
# ============================================================

@st.cache_data
def load_data_from_files():
    datasets = []
    
    # Almaty
    almaty_raw = read_csv_safe("air_quality_data.csv")
    if almaty_raw is not None:
        almaty = standardize_dataset(almaty_raw, forced_city="Almaty")
        if almaty is not None:
            datasets.append(almaty)
        else:
            st.warning("⚠️ air_quality_data.csv found but could not be parsed. Check column names.")
    else:
        st.info("ℹ️ air_quality_data.csv not found in the current directory.")
    
    # Astana
    astana_raw = read_csv_safe("14.csv")
    if astana_raw is not None:
        astana = standardize_dataset(astana_raw, forced_city="Astana")
        if astana is not None:
            datasets.append(astana)
        else:
            st.warning("⚠️ 14.csv found but could not be parsed. Check column names.")
    else:
        st.info("ℹ️ 14.csv not found in the current directory.")
    
    if not datasets:
        return None
    
    combined = pd.concat(datasets, ignore_index=True)
    combined = combined.drop_duplicates(subset=["time", "city", "pm25"])
    return combined

# Try to load from files
df_full = load_data_from_files()

# If no data, show upload interface
if df_full is None:
    st.warning("No data files found. Please upload your CSV files below.")
    
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Upload Almaty data (air_quality_data.csv)", type=['csv'])
    with col2:
        file2 = st.file_uploader("Upload Astana data (14.csv)", type=['csv'])
    
    if file1 is not None or file2 is not None:
        datasets = []
        if file1 is not None:
            try:
                df1 = pd.read_csv(file1)
                std1 = standardize_dataset(df1, forced_city="Almaty")
                if std1 is not None:
                    datasets.append(std1)
                    st.success("✅ Almaty data loaded successfully.")
                else:
                    st.error("❌ Could not parse Almaty file. Check column names (need time, PM2.5).")
            except Exception as e:
                st.error(f"Error reading Almaty file: {e}")
        if file2 is not None:
            try:
                df2 = pd.read_csv(file2)
                std2 = standardize_dataset(df2, forced_city="Astana")
                if std2 is not None:
                    datasets.append(std2)
                    st.success("✅ Astana data loaded successfully.")
                else:
                    st.error("❌ Could not parse Astana file. Check column names (need time, PM2.5).")
            except Exception as e:
                st.error(f"Error reading Astana file: {e}")
        
        if datasets:
            df_full = pd.concat(datasets, ignore_index=True)
            df_full = df_full.drop_duplicates(subset=["time", "city", "pm25"])
            st.success(f"✅ Total records loaded: {len(df_full):,}")
        else:
            st.error("❌ No valid data could be loaded. Please check your files.")
            st.stop()
    else:
        st.stop()

# If still no data, stop
if df_full is None or df_full.empty:
    st.error("❌ No data available. Please upload valid CSV files.")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📁 Dataset")
st.sidebar.success(f"Loaded records: {len(df_full):,}")
st.sidebar.write(f"Cities: {df_full['city'].nunique()}")

st.sidebar.header("📍 Select City")
cities = sorted(df_full["city"].dropna().unique())
selected_city = st.sidebar.selectbox("City:", cities)

# ============================================================
# CITY DATA
# ============================================================

df_city = df_full[df_full["city"] == selected_city].copy()
df_city = df_city.sort_values("time")

if len(df_city) < 50:
    st.error(f"❌ Not enough data for **{selected_city}**.")
    st.write(f"Available records: {len(df_city)}")
    st.stop()

# ============================================================
# FEATURES
# ============================================================

df_city["hour"] = df_city["time"].dt.hour
df_city["dayofweek"] = df_city["time"].dt.dayofweek
df_city["month"] = df_city["time"].dt.month
df_city["day"] = df_city["time"].dt.day

for lag in [1, 3, 6, 12, 24]:
    df_city[f"pm25_lag_{lag}"] = df_city["pm25"].shift(lag)

df_model = df_city.dropna().copy()

if len(df_model) < 50:
    st.error(f"❌ After creating lag features, only {len(df_model)} records remain.")
    st.stop()

features = ["hour", "dayofweek", "month", "day",
            "pm25_lag_1", "pm25_lag_3", "pm25_lag_6",
            "pm25_lag_12", "pm25_lag_24"]

# ============================================================
# TRAIN MODEL
# ============================================================

X = df_model[features]
y = df_model["pm25"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.subheader(f"📊 {selected_city}")
c1, c2, c3 = st.columns(3)
c1.metric("R² Score", f"{r2:.4f}")
c2.metric("MAE", f"{mae:.2f} µg/m³")
c3.metric("Records", f"{len(df_city):,}")

st.caption(f"Data period: {df_city['time'].min().strftime('%Y-%m-%d')} → {df_city['time'].max().strftime('%Y-%m-%d')}")
st.markdown("---")

# ============================================================
# FORECAST
# ============================================================

def forecast_future(model, df, features, hours_ahead=3):
    work = df.copy()
    forecasts = []
    last_time = work["time"].iloc[-1]
    pm_values = list(work["pm25"].iloc[-24:].values)
    
    for step in range(1, hours_ahead + 1):
        future_time = last_time + timedelta(hours=step)
        row = {
            "hour": future_time.hour,
            "dayofweek": future_time.weekday(),
            "month": future_time.month,
            "day": future_time.day
        }
        row["pm25_lag_1"] = pm_values[-1]
        row["pm25_lag_3"] = pm_values[-3] if len(pm_values) >= 3 else pm_values[-1]
        row["pm25_lag_6"] = pm_values[-6] if len(pm_values) >= 6 else pm_values[-1]
        row["pm25_lag_12"] = pm_values[-12] if len(pm_values) >= 12 else pm_values[-1]
        row["pm25_lag_24"] = pm_values[-24] if len(pm_values) >= 24 else pm_values[-1]
        
        X_future = pd.DataFrame([row])[features]
        pred = model.predict(X_future)[0]
        pred = max(0, float(pred))
        forecasts.append(pred)
        pm_values.append(pred)
    
    return forecasts

last_actual = float(df_city["pm25"].iloc[-1])
last_time = df_city["time"].iloc[-1]
forecast_values = forecast_future(model, df_city, features, hours_ahead=3)

# ============================================================
# EARLY WARNING
# ============================================================

st.header("⚠️ Early Warning System")
WARNING_THRESHOLD = 50
st.caption(f"Warning threshold: {WARNING_THRESHOLD} µg/m³ (WHO)")

if last_actual < 50:
    status = "🟢 Good"
else:
    status = "🔴 Unhealthy"

st.subheader(f"{status} — Current PM2.5: {last_actual:.1f} µg/m³")

cols = st.columns(3)
for i, val in enumerate(forecast_values):
    is_warning = val >= WARNING_THRESHOLD
    future_time = last_time + timedelta(hours=i+1)
    with cols[i]:
        st.metric(
            label=f"{'🔴' if is_warning else '🟢'} +{i+1}h ({future_time.strftime('%H:%M')})",
            value=f"{val:.1f} µg/m³",
            delta=f"{val - last_actual:+.1f}"
        )
        if is_warning:
            st.warning("⚠️ Exceeds threshold")
        else:
            st.success("✅ OK")

if max(forecast_values) >= WARNING_THRESHOLD:
    st.error("🚨 WARNING: PM2.5 is forecast to exceed 50 µg/m³ in the next few hours.")
    st.info("💡 Recommendations: Wear a mask, stay indoors, close windows, use air purifier.")
else:
    st.success("✅ No warning: PM2.5 is expected to remain below the threshold.")

st.markdown("---")

# ============================================================
# PLOTS
# ============================================================

st.header("📈 PM2.5 Forecast — Next 3 Hours")
fig, ax = plt.subplots(figsize=(12, 5))
history = df_city["pm25"].iloc[-24:].values
history_x = np.arange(len(history))
forecast_x = np.arange(len(history), len(history) + len(forecast_values))
ax.plot(history_x, history, label="Historical", linewidth=2)
ax.plot(forecast_x, forecast_values, "--o", label="Forecast", linewidth=2)
ax.axhline(WARNING_THRESHOLD, linestyle=":", linewidth=2, label="Warning threshold")
ax.axvline(len(history) - 1, linestyle="--", linewidth=1, label="Now")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title(f"{selected_city} — PM2.5 Forecast")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

st.markdown("---")
st.header("📊 Model Performance")
fig2, ax2 = plt.subplots(figsize=(12, 5))
n = min(150, len(y_test))
ax2.plot(range(n), y_test.iloc[:n].values, label="Actual", linewidth=2)
ax2.plot(range(n), y_pred[:n], label="Predicted", linewidth=2)
ax2.set_xlabel("Test observations")
ax2.set_ylabel("PM2.5 (µg/m³)")
ax2.set_title(f"{selected_city} — Actual vs Predicted")
ax2.legend()
ax2.grid(alpha=0.3)
st.pyplot(fig2)
plt.close(fig2)

st.markdown("---")
st.header("🔍 Feature Importance")
importance_df = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
importance_df = importance_df.sort_values("Importance", ascending=True)
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.barh(importance_df["Feature"], importance_df["Importance"])
ax3.set_xlabel("Importance")
ax3.set_ylabel("Feature")
ax3.set_title(f"{selected_city} — Feature Importance")
ax3.grid(axis="x", alpha=0.3)
st.pyplot(fig3)
plt.close(fig3)

st.markdown("---")
with st.expander("🔎 View latest measurements"):
    st.dataframe(df_city[["time", "city", "pm25"]].tail(20), use_container_width=True)

st.markdown("---")
st.caption("Air Quality Prediction System © 2026 Nurikamal Bolatbay")
