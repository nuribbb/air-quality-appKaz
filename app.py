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

st.set_page_config(page_title="Air Quality Prediction / Ауа сапасы / Качество воздуха", layout="wide")

# ============================================================
# LANGUAGE SELECTION
# ============================================================

if 'lang' not in st.session_state:
    st.session_state.lang = 'kaz'

lang = st.sidebar.radio("🌐 Тіл / Язык / Language", ['🇰🇿 Қазақша', '🇷🇺 Русский', '🇬🇧 English'], index=0)

if lang == '🇰🇿 Қазақша':
    LANG = 'kaz'
elif lang == '🇷🇺 Русский':
    LANG = 'rus'
else:
    LANG = 'eng'

# ============================================================
# TRANSLATIONS
# ============================================================

T = {}

if LANG == 'kaz':
    T['title'] = "🌍 Ауа сапасын болжау жүйесі"
    T['author'] = "👤 **Автор:** Nurikamal Bolatbay"
    T['dataset'] = "📁 Деректер"
    T['records'] = "Жазбалар: {:,}"
    T['cities'] = "Қалалар: {}"
    T['select_city'] = "📍 Қаланы таңдаңыз"
    T['city'] = "Қала:"
    T['r2'] = "R²"
    T['mae'] = "Орташа қате (MAE)"
    T['records_short'] = "Жазбалар"
    T['period'] = "Кезең: {} → {}"
    T['warning'] = "⚠️ Ерте ескерту жүйесі"
    T['threshold'] = "Ескерту шегі: {} µg/m³ (ДДҰ)"
    T['current'] = "Ағымдағы жағдай"
    T['pm25'] = "PM2.5: {:.1f} µg/m³ — {}"
    T['forecast_title'] = "Келесі 3 сағатқа болжам:"
    T['exceeds'] = "⚠️ Шектен асады!"
    T['ok'] = "✅ Қалыпты"
    T['warning_alert'] = "🚨 **ЕСКЕРТУ:** PM2.5 мөлшері келесі сағаттарда 50 µg/m³ асуы мүмкін. Сактық шараларын қолданыңыз."
    T['no_warning'] = "✅ Ескерту жоқ: PM2.5 деңгейі қауіпсіз шекте қалады."
    T['rec_good'] = "💡 **Ұсыныстар:** Сыртта белсенді болыңыз, бірақ ауа сапасының өзгерістерін бақылаңыз."
    T['rec_sensitive'] = "💡 **Ұсыныстар:** Сезімтал топтар (балалар, қарттар, тыныс алу аурулары бар адамдар) ұзақ уақыт сыртта болмауы керек."
    T['rec_bad'] = "💡 **Ұсыныстар:**\n- Маска киіңіз (N95 ұсынылады)\n- Үйде қалыңыз\n- Барлық терезелерді жабыңыз\n- Ауа тазартқышты қосыңыз\n- Сыртта дене жаттығуларынан аулақ болыңыз\n- Есіктерді тығыз жабыңыз\n- Осал топтарды (балалар, қарттар, тыныс алу аурулары бар адамдар) бақылаңыз"
    T['forecast_plot'] = "📈 PM2.5 болжамы — келесі 3 сағат"
    T['history'] = "Тарихи деректер"
    T['now'] = "Қазір"
    T['forecast'] = "Болжам"
    T['threshold_line'] = "Қауіп шегі"
    T['model_perf'] = "📊 Модель сапасы"
    T['actual'] = "Нақты мәндер"
    T['predicted'] = "Болжанған мәндер"
    T['feature_importance'] = "🔍 Белгілердің маңыздылығы"
    T['feature'] = "Белгі"
    T['importance'] = "Маңыздылық"
    T['latest'] = "🔎 Соңғы өлшемдер"

elif LANG == 'rus':
    T['title'] = "🌍 Система прогнозирования качества воздуха"
    T['author'] = "👤 **Автор:** Nurikamal Bolatbay"
    T['dataset'] = "📁 Данные"
    T['records'] = "Записей: {:,}"
    T['cities'] = "Городов: {}"
    T['select_city'] = "📍 Выберите город"
    T['city'] = "Город:"
    T['r2'] = "R²"
    T['mae'] = "Средняя ошибка (MAE)"
    T['records_short'] = "Записей"
    T['period'] = "Период: {} → {}"
    T['warning'] = "⚠️ Система раннего предупреждения"
    T['threshold'] = "Порог предупреждения: {} µg/m³ (ВОЗ)"
    T['current'] = "Текущее состояние"
    T['pm25'] = "PM2.5: {:.1f} µg/m³ — {}"
    T['forecast_title'] = "Прогноз на следующие 3 часа:"
    T['exceeds'] = "⚠️ Превышает порог!"
    T['ok'] = "✅ Норма"
    T['warning_alert'] = "🚨 **ВНИМАНИЕ:** Прогнозируется превышение безопасного порога (50 µg/m³) в ближайшие часы. Примите меры."
    T['no_warning'] = "✅ Предупреждений нет: уровень PM2.5 останется ниже порога."
    T['rec_good'] = "💡 **Рекомендации:** Наслаждайтесь активностью на улице, но следите за изменениями качества воздуха."
    T['rec_sensitive'] = "💡 **Рекомендации:** Чувствительным группам (дети, пожилые, люди с респираторными заболеваниями) следует ограничить длительное пребывание на улице."
    T['rec_bad'] = "💡 **Рекомендации:**\n- Используйте маску (N95 рекомендуется)\n- Оставайтесь дома\n- Закройте все окна\n- Включите очиститель воздуха\n- Избегайте физической активности на улице\n- Держите двери закрытыми\n- Следите за уязвимыми группами (дети, пожилые, люди с респираторными заболеваниями)"
    T['forecast_plot'] = "📈 Прогноз PM2.5 — следующие 3 часа"
    T['history'] = "Исторические данные"
    T['now'] = "Сейчас"
    T['forecast'] = "Прогноз"
    T['threshold_line'] = "Порог опасности"
    T['model_perf'] = "📊 Качество модели"
    T['actual'] = "Фактические значения"
    T['predicted'] = "Предсказанные значения"
    T['feature_importance'] = "🔍 Важность признаков"
    T['feature'] = "Признак"
    T['importance'] = "Важность"
    T['latest'] = "🔎 Последние измерения"

else:  # English
    T['title'] = "🌍 Air Quality Prediction System"
    T['author'] = "👤 **Author:** Nurikamal Bolatbay"
    T['dataset'] = "📁 Dataset"
    T['records'] = "Records: {:,}"
    T['cities'] = "Cities: {}"
    T['select_city'] = "📍 Select City"
    T['city'] = "City:"
    T['r2'] = "R² Score"
    T['mae'] = "Mean Absolute Error (MAE)"
    T['records_short'] = "Records"
    T['period'] = "Period: {} → {}"
    T['warning'] = "⚠️ Early Warning System"
    T['threshold'] = "Warning threshold: {} µg/m³ (WHO)"
    T['current'] = "Current Status"
    T['pm25'] = "PM2.5: {:.1f} µg/m³ — {}"
    T['forecast_title'] = "Forecast for the next 3 hours:"
    T['exceeds'] = "⚠️ Exceeds threshold!"
    T['ok'] = "✅ OK"
    T['warning_alert'] = "🚨 **WARNING:** PM2.5 is forecast to exceed the safe threshold (50 µg/m³) within the next few hours. Take precautions."
    T['no_warning'] = "✅ No warning: PM2.5 is expected to remain below the threshold."
    T['rec_good'] = "💡 **Recommendations:** Enjoy outdoor activities, but stay aware of any changes in air quality."
    T['rec_sensitive'] = "💡 **Recommendations:** Sensitive groups (children, elderly, people with respiratory conditions) should limit prolonged outdoor exertion."
    T['rec_bad'] = "💡 **Recommendations:**\n- Wear a mask (N95 recommended)\n- Stay indoors\n- Close all windows\n- Use an air purifier\n- Avoid outdoor physical activities\n- Keep doors sealed\n- Monitor vulnerable groups (children, elderly, those with respiratory conditions)"
    T['forecast_plot'] = "📈 PM2.5 Forecast — Next 3 Hours"
    T['history'] = "Historical Data"
    T['now'] = "Now"
    T['forecast'] = "Forecast"
    T['threshold_line'] = "Warning threshold"
    T['model_perf'] = "📊 Model Performance"
    T['actual'] = "Actual"
    T['predicted'] = "Predicted"
    T['feature_importance'] = "🔍 Feature Importance"
    T['feature'] = "Feature"
    T['importance'] = "Importance"
    T['latest'] = "🔎 View latest measurements"

# ============================================================
# LEVELS
# ============================================================

PM25_LEVELS = [
    (0, 12, "Good", "Қалыпты", "Хорошо", "🟢"),
    (12.1, 35.4, "Moderate", "Орташа", "Умеренно", "🟡"),
    (35.5, 55.4, "Unhealthy for Sensitive Groups", "Сезімтал топтар үшін қауіпті", "Неблагоприятно для чувствительных групп", "🟠"),
    (55.5, 150.4, "Unhealthy", "Қауіпті", "Неблагоприятно", "🔴"),
    (150.5, 250.4, "Very Unhealthy", "Өте қауіпті", "Очень неблагоприятно", "🟣"),
    (250.5, float('inf'), "Hazardous", "Қатерлі", "Опасно", "⚫")
]

def get_level(pm25):
    for low, high, eng, kaz, rus, emoji in PM25_LEVELS:
        if low <= pm25 <= high:
            if LANG == 'kaz':
                return kaz, emoji
            elif LANG == 'rus':
                return rus, emoji
            else:
                return eng, emoji
    return "Unknown", "❓"

# ============================================================
# READ CSV (AUTO-DETECT COLUMNS)
# ============================================================

def read_csv_safe(filename):
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
        return pd.DataFrame()
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    
    time_col = find_column(df, ["time", "datetime", "date", "timestamp", "last_updated"])
    if time_col is None:
        possible = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        if possible.notna().sum() > len(df) * 0.5:
            time_col = df.columns[0]
    if time_col is None:
        return pd.DataFrame()
    
    df["time"] = pd.to_datetime(df[time_col], errors='coerce', utc=True)
    try:
        df["time"] = df["time"].dt.tz_localize(None)
    except:
        pass
    
    pm_col = find_column(df, ["pm2.5", "pm25", "pm2_5", "pm_2_5", "pm 2.5", "pm2"])
    if pm_col is None:
        return pd.DataFrame()
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
    return df

# ============================================================
# LOAD DATA (UPLOAD VIA INTERFACE IF FILES NOT FOUND)
# ============================================================

@st.cache_data
def load_data_from_files(almaty_df, astana_df):
    datasets = []
    if almaty_df is not None:
        almaty_std = standardize_dataset(almaty_df, forced_city="Алматы")
        if not almaty_std.empty:
            datasets.append(almaty_std)
    if astana_df is not None:
        astana_std = standardize_dataset(astana_df, forced_city="Астана")
        if not astana_std.empty:
            datasets.append(astana_std)
    if not datasets:
        return pd.DataFrame()
    combined = pd.concat(datasets, ignore_index=True)
    combined = combined.drop_duplicates(subset=["time", "city", "pm25"])
    return combined

# ============================================================
# INTERFACE
# ============================================================

st.title(T['title'])
st.markdown(T['author'])
st.markdown("---")

# Try to load existing files
almaty_file = read_csv_safe("air_quality_data.csv")
astana_file = read_csv_safe("14.csv")

if almaty_file is not None or astana_file is not None:
    # If files exist in the directory, use them
    almaty_df = almaty_file if almaty_file is not None else None
    astana_df = astana_file if astana_file is not None else None
    df_full = load_data_from_files(almaty_df, astana_df)
    if not df_full.empty:
        st.success("✅ Data loaded from local files.")
    else:
        df_full = pd.DataFrame()
else:
    df_full = pd.DataFrame()

# If no data loaded, show upload interface
if df_full.empty:
    st.info("📂 **Upload your data files to start analysis.**")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_almaty = st.file_uploader("Upload Алматы data (air_quality_data.csv)", type=['csv'])
    with col2:
        uploaded_astana = st.file_uploader("Upload Астана data (14.csv)", type=['csv'])
    
    if uploaded_almaty is not None or uploaded_astana is not None:
        almaty_df = pd.read_csv(uploaded_almaty) if uploaded_almaty is not None else None
        astana_df = pd.read_csv(uploaded_astana) if uploaded_astana is not None else None
        df_full = load_data_from_files(almaty_df, astana_df)
        if not df_full.empty:
            st.success("✅ Data loaded from uploaded files.")
        else:
            st.error("❌ Could not parse uploaded files. Please check format.")
    else:
        st.stop()

if df_full.empty:
    st.error("❌ No data available. Please upload files.")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(T['dataset'])
st.sidebar.success(T['records'].format(len(df_full)))
st.sidebar.write(T['cities'].format(df_full['city'].nunique()))

st.sidebar.header(T['select_city'])
cities = sorted(df_full["city"].dropna().unique())
for city in ["Астана", "Алматы"]:
    if city in cities:
        cities.remove(city)
        cities.insert(0, city)
selected_city = st.sidebar.selectbox(T['city'], cities)

# ============================================================
# CITY DATA
# ============================================================

df_city = df_full[df_full["city"] == selected_city].copy()
df_city = df_city.sort_values("time")
df_city.rename(columns={'pm2_5': 'pm25'}, inplace=True)

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
    st.error(f"❌ After lag features, only {len(df_model)} records remain.")
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
c1.metric(T['r2'], f"{r2:.4f}")
c2.metric(T['mae'], f"{mae:.2f} µg/m³")
c3.metric(T['records_short'], f"{len(df_city):,}")

st.caption(T['period'].format(
    df_city['time'].min().strftime('%Y-%m-%d'),
    df_city['time'].max().strftime('%Y-%m-%d')
))
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
# EARLY WARNING SYSTEM
# ============================================================

st.header(T['warning'])
WARNING_THRESHOLD = 50
st.caption(T['threshold'].format(WARNING_THRESHOLD))

current_label, current_emoji = get_level(last_actual)
st.subheader(f"{current_emoji} {T['current']}: {current_label}")
st.write(T['pm25'].format(last_actual, current_label))

st.write("---")
st.write("**" + T['forecast_title'] + "**")

cols = st.columns(3)
for i, val in enumerate(forecast_values):
    label, emoji = get_level(val)
    future_time = last_time + timedelta(hours=i+1)
    with cols[i]:
        st.metric(
            label=f"{emoji} +{i+1}ч ({future_time.strftime('%H:%M')})",
            value=f"{val:.1f} µg/m³",
            delta=f"{val - last_actual:+.1f}"
        )
        st.caption(f"**{label}**")
        if val >= WARNING_THRESHOLD:
            st.warning(T['exceeds'])
        else:
            st.success(T['ok'])

max_forecast = max(forecast_values)
if max_forecast >= WARNING_THRESHOLD:
    st.error(T['warning_alert'])
else:
    st.success(T['no_warning'])

max_label, _ = get_level(max_forecast)

if max_label in ["Неблагоприятно", "Очень неблагоприятно", "Опасно", "Қауіпті", "Өте қауіпті", "Қатерлі", "Unhealthy", "Very Unhealthy", "Hazardous"]:
    st.info(T['rec_bad'])
elif max_label in ["Неблагоприятно для чувствительных групп", "Сезімтал топтар үшін қауіпті", "Unhealthy for Sensitive Groups"]:
    st.info(T['rec_sensitive'])
else:
    st.info(T['rec_good'])

st.markdown("---")

# ============================================================
# PLOTS
# ============================================================

st.header(T['forecast_plot'])
fig, ax = plt.subplots(figsize=(12, 5))
history = df_city["pm25"].iloc[-24:].values
history_x = np.arange(len(history))
forecast_x = np.arange(len(history), len(history) + len(forecast_values))
ax.plot(history_x, history, label=T['history'], linewidth=2)
ax.plot(forecast_x, forecast_values, "--o", label=T['forecast'], linewidth=2)
ax.axhline(WARNING_THRESHOLD, linestyle=":", linewidth=2, label=T['threshold_line'])
ax.axvline(len(history) - 1, linestyle="--", linewidth=1, label=T['now'])
ax.set_xlabel("Time (hours)")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title(f"{selected_city}: PM2.5 Forecast")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig, clear_figure=True)
plt.close(fig)

st.markdown("---")
st.header(T['model_perf'])
fig2, ax2 = plt.subplots(figsize=(12, 5))
n = min(150, len(y_test))
ax2.plot(range(n), y_test.iloc[:n].values, label=T['actual'], linewidth=2)
ax2.plot(range(n), y_pred[:n], label=T['predicted'], linewidth=2)
ax2.set_xlabel("Test observations")
ax2.set_ylabel("PM2.5 (µg/m³)")
ax2.set_title(f"{selected_city}: Actual vs Predicted")
ax2.legend()
ax2.grid(alpha=0.3)
st.pyplot(fig2, clear_figure=True)
plt.close(fig2)

st.markdown("---")
st.header(T['feature_importance'])
importance_df = pd.DataFrame({
    T['feature']: features,
    T['importance']: model.feature_importances_
})
importance_df = importance_df.sort_values(T['importance'], ascending=True)

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.barh(importance_df[T['feature']], importance_df[T['importance']])
ax3.set_xlabel(T['importance'])
ax3.set_ylabel(T['feature'])
ax3.set_title(f"{selected_city}: {T['feature_importance']}")
ax3.grid(axis="x", alpha=0.3)
st.pyplot(fig3, clear_figure=True)
plt.close(fig3)

st.markdown("---")
with st.expander(T['latest']):
    st.dataframe(df_city[["time", "city", "pm25"]].tail(20), use_container_width=True)

st.markdown("---")
st.caption("Air Quality Prediction System ©️ 2026 Nurikamal Bolatbay")
