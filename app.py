import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import timedelta
import requests
import io

st.set_page_config(page_title="Air Quality Prediction", layout="wide")
st.title("🌍 Air Quality Prediction System")
st.markdown("👤 **Author:** Nurikamal Bolatbay")
st.markdown("---")

# ============================================================
# READ DATA FROM GOOGLE DRIVE
# ============================================================

def read_csv_from_drive(file_id):
    """Download CSV from Google Drive using file ID."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return pd.read_csv(io.StringIO(response.text))
        else:
            return None
    except:
        return None

# Try to get file IDs from session state or from input
if 'almaty_id' not in st.session_state:
    st.session_state.almaty_id = ""
if 'astana_id' not in st.session_state:
    st.session_state.astana_id = ""

# Show inputs for Google Drive file IDs
st.sidebar.header("📁 Google Drive Links")
almaty_id = st.sidebar.text_input("Almaty file ID (air_quality_data.csv)", value=st.session_state.almaty_id)
astana_id = st.sidebar.text_input("Astana file ID (14.csv)", value=st.session_state.astana_id)

if almaty_id:
    st.session_state.almaty_id = almaty_id
if astana_id:
    st.session_state.astana_id = astana_id

# Load data if IDs are provided
df_full = None
if st.session_state.almaty_id or st.session_state.astana_id:
    datasets = []
    if st.session_state.almaty_id:
        with st.spinner("Loading Almaty data from Google Drive..."):
            df_almaty_raw = read_csv_from_drive(st.session_state.almaty_id)
            if df_almaty_raw is not None:
                df_almaty_raw['city'] = 'Almaty'
                datasets.append(df_almaty_raw)
                st.success("✅ Almaty data loaded from Google Drive")
            else:
                st.error("❌ Failed to load Almaty data. Check file ID.")
    if st.session_state.astana_id:
        with st.spinner("Loading Astana data from Google Drive..."):
            df_astana_raw = read_csv_from_drive(st.session_state.astana_id)
            if df_astana_raw is not None:
                df_astana_raw['city'] = 'Astana'
                datasets.append(df_astana_raw)
                st.success("✅ Astana data loaded from Google Drive")
            else:
                st.error("❌ Failed to load Astana data. Check file ID.")
    if datasets:
        df_full = pd.concat(datasets, ignore_index=True)
    else:
        st.warning("No data loaded. Please check file IDs.")
else:
    st.info("ℹ️ Enter Google Drive file IDs in the sidebar to load real data.")

# If no data, show fallback message
if df_full is None or df_full.empty:
    st.warning("No data available. Please enter file IDs in the sidebar.")
    # You could also provide a fallback synthetic option here, but we'll just stop.
    st.stop()

# ============================================================
# PROCESS DATA (auto-detect columns)
# ============================================================

def find_column(df, possible_names):
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

def standardize_dataset(df, forced_city=None):
    if df is None or df.empty:
        return None
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    
    time_col = find_column(df, ["time", "datetime", "date", "timestamp", "last_updated"])
    if time_col is None:
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
    
    pm_col = find_column(df, ["pm2.5", "pm25", "pm2_5", "pm_2_5", "pm 2.5", "pm2"])
    if pm_col is None:
        return None
    df["pm25"] = pd.to_numeric(df[pm_col], errors='coerce')
    
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

# Standardize each city separately
city_data = []
for city in df_full['city'].unique():
    df_city = df_full[df_full['city'] == city]
    std = standardize_dataset(df_city, forced_city=city)
    if std is not None:
        city_data.append(std)

if not city_data:
    st.error("Could not standardize data. Check column names.")
    st.stop()

df_full = pd.concat(city_data, ignore_index=True)
df_full = df_full.drop_duplicates(subset=["time", "city", "pm25"])

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📁 Dataset")
st.sidebar.success(f"Records: {len(df_full):,}")
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
    st.error(f"Not enough data for {selected_city}. Available: {len(df_city)}")
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
    st.error(f"After lag features, only {len(df_model)} records remain.")
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

st.caption(f"Period: {df_city['time'].min().strftime('%Y-%m-%d')} → {df_city['time'].max().strftime('%Y-%m-%d')}")
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
st.caption(f"Threshold: {WARNING_THRESHOLD} µg/m³ (WHO)")

status = "🟢 Good" if last_actual < 50 else "🔴 Unhealthy"
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
    st.error("🚨 WARNING: PM2.5 will exceed 50 µg/m³.")
    st.info("💡 Recommendations: Wear mask, stay indoors, close windows, use purifier.")
else:
    st.success("✅ No warning: Air quality is safe.")

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
ax.axhline(WARNING_THRESHOLD, linestyle=":", linewidth=2, label="Threshold")
ax.axvline(len(history) - 1, linestyle="--", linewidth=1, label="Now")
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
ax3.grid(axis="x", alpha=0.3)
st.pyplot(fig3)
plt.close(fig3)

st.markdown("---")
with st.expander("🔎 View latest measurements"):
    st.dataframe(df_city[["time", "city", "pm25"]].tail(20), use_container_width=True)

st.markdown("---")
st.caption("Air Quality Prediction System © 2026 Nurikamal Bolatbay")
