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
# GOOGLE DRIVE FILE IDs
# ============================================================

ALMATY_ID = "1NKll16FnrW9i9D48n5k0UpSzN6HQcBQQ"
ASTANA_ID = "1vx5cGXYwp3NY2kVQnClzRCHNUgof5KW"

def load_csv_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            return pd.read_csv(io.StringIO(r.text))
        else:
            return None
    except:
        return None

# ============================================================
# SIDEBAR – CITY SELECTION
# ============================================================

st.sidebar.header("📍 Select City")
city_choice = st.sidebar.selectbox("City:", ["Almaty", "Astana"])

if city_choice == "Almaty":
    file_id = ALMATY_ID
    city_name = "Almaty"
else:
    file_id = ASTANA_ID
    city_name = "Astana"

st.sidebar.info(f"Loading data for **{city_name}** from Google Drive")

# ============================================================
# LOAD DATA FOR SELECTED CITY
# ============================================================

with st.spinner(f"Downloading {city_name} data from Google Drive..."):
    df_raw = load_csv_from_drive(file_id)

if df_raw is None:
    st.error(f"❌ Failed to load {city_name} data. Check file ID or internet connection.")
    st.stop()

st.success(f"✅ {city_name} data loaded: {len(df_raw):,} rows")

# ============================================================
# AUTO-DETECT COLUMNS AND STANDARDIZE
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

def standardize(df, forced_city=None):
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
    
    df["city"] = forced_city if forced_city else "Unknown"
    
    df = df[["time", "city", "pm25"]].copy()
    df = df.dropna(subset=["time", "pm25"])
    df = df[(df["pm25"] >= 0) & (df["pm25"] <= 1000)]
    df = df.sort_values("time")
    if df.empty:
        return None
    return df

df_std = standardize(df_raw, forced_city=city_name)

if df_std is None:
    st.error(f"❌ Could not parse {city_name} data. Check column names.")
    st.write("Columns found:", df_raw.columns.tolist())
    st.stop()

st.success(f"✅ Standardized {city_name} data: {len(df_std):,} records")

# ============================================================
# DATA STATS
# ============================================================

st.sidebar.header("📁 Dataset")
st.sidebar.success(f"Records: {len(df_std):,}")
st.sidebar.write(f"City: {city_name}")

# ============================================================
# PREPARE DATA FOR MODEL
# ============================================================

df_city = df_std.copy()
df_city = df_city.sort_values("time")

if len(df_city) < 50:
    st.error(f"Not enough data for {city_name}. Available: {len(df_city)}")
    st.stop()

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

st.subheader(f"📊 {city_name}")
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
