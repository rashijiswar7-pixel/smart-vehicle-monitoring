from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "cleaned"
MODEL_PATH = BASE_DIR / "models" / "vehicle_lstm.keras"
SCALER_PATH = BASE_DIR / "models" / "scaler_data.npz"
WINDOW_SIZE = 10


# ============================================================
# 2. PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Smart Vehicle Health Monitoring",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Smart Vehicle Health Monitoring System")
st.markdown(
    "OBD-II sensor analysis and experimental LSTM-based anomaly detection."
)


# ============================================================
# 3. LOAD DATA AND MODEL
# ============================================================

@st.cache_data
def load_dataset():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data folder not found: {DATA_DIR}. "
            "Check that your cleaned CSV files are inside data/cleaned/."
        )

    files = sorted(DATA_DIR.glob("*_cleaned.csv"))
    if not files:
        # Allow a CSV in data/cleaned even if it has a different suffix.
        files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            "No CSV files were found in data/cleaned/."
        )

    # Keep using the first sorted file, consistent with the previous app.
    selected_file = files[0]
    frame = pd.read_csv(selected_file, low_memory=False)
    return frame, selected_file.name


@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. Run model.py first."
        )
    return load_model(MODEL_PATH)


def load_saved_scaler():
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler file not found: {SCALER_PATH}. Run model.py first."
        )

    saved = np.load(SCALER_PATH, allow_pickle=False)
    required_keys = {
        "feature_names", "threshold", "scale", "min",
        "data_min", "data_max"
    }
    missing = required_keys.difference(saved.files)
    if missing:
        raise ValueError(
            "The saved scaler file is missing these values: "
            + ", ".join(sorted(missing))
            + ". The scaler-saving section in model.py may need updating."
        )

    feature_names = saved["feature_names"].astype(str).tolist()
    scaler = MinMaxScaler()
    scaler.scale_ = saved["scale"]
    scaler.min_ = saved["min"]
    scaler.data_min_ = saved["data_min"]
    scaler.data_max_ = saved["data_max"]
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    scaler.n_features_in_ = len(feature_names)
    scaler.n_samples_seen_ = 1
    threshold = float(saved["threshold"])
    saved.close()
    return feature_names, scaler, threshold


# ============================================================
# 4. PREPARE DATA AND RUN ANOMALY DETECTION
# ============================================================

def prepare_sensor_data(frame, feature_names):
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(
            "The dataset does not contain all model features: "
            + ", ".join(missing)
        )

    sensors = frame[feature_names].copy()

    for column in sensors.columns:
        # Accommodate numeric values stored as text, including decimal commas.
        if sensors[column].dtype == object:
            text_values = (
                sensors[column]
                .astype(str)
                .str.strip()
                .str.replace("%", "", regex=False)
            )
            # Only replace comma with decimal point when no decimal point exists.
            text_values = text_values.map(
                lambda value: value.replace(",", ".")
                if "," in value and "." not in value
                else value
            )
            sensors[column] = pd.to_numeric(text_values, errors="coerce")
        else:
            sensors[column] = pd.to_numeric(
                sensors[column], errors="coerce"
            )

    sensors = sensors.replace([np.inf, -np.inf], np.nan)
    sensors = sensors.interpolate(limit_direction="both")
    sensors = sensors.ffill().bfill()

    # Remove rows that still contain invalid values.
    valid_mask = sensors.notna().all(axis=1)
    sensors = sensors.loc[valid_mask]

    return sensors, valid_mask


def analyse_readings(frame, model, scaler, feature_names, threshold):
    sensors, valid_mask = prepare_sensor_data(frame, feature_names)

    if len(sensors) <= WINDOW_SIZE:
        raise ValueError(
            f"Only {len(sensors)} valid rows are available; "
            f"at least {WINDOW_SIZE + 1} are needed."
        )

    scaled = scaler.transform(sensors)
    X = np.asarray([
        scaled[i - WINDOW_SIZE:i]
        for i in range(WINDOW_SIZE, len(scaled))
    ])
    actual = scaled[WINDOW_SIZE:]
    predicted = model.predict(X, verbose=0)

    errors = np.mean(np.square(actual - predicted), axis=1)

    results = sensors.iloc[WINDOW_SIZE:].copy()
    results["Anomaly Error"] = errors
    results["Anomaly"] = results["Anomaly Error"] > threshold

    # Sequential reading numbers make the horizontal axis readable.
    results["Reading Number"] = np.arange(1, len(results) + 1)

    # Add timestamps as table information only; never use them as the chart axis.
    if "TIMESTAMP" in frame.columns:
        timestamps = pd.to_datetime(frame.loc[results.index, "TIMESTAMP"],
                                    errors="coerce")
        results["Timestamp"] = timestamps.to_numpy()

    return results


try:
    raw_df, dataset_name = load_dataset()
    model = load_trained_model()
    feature_names, scaler, threshold = load_saved_scaler()
    results = analyse_readings(
        raw_df, model, scaler, feature_names, threshold
    )
except Exception as exc:
    st.error(f"Unable to analyse the dataset: {exc}")
    st.info(
        "Check that the cleaned CSV, vehicle_lstm.keras, and "
        "scaler_data.npz are in the expected folders."
    )
    st.stop()


# ============================================================
# 5. SUMMARY METRICS
# ============================================================

total_readings = len(results)
flagged_count = int(results["Anomaly"].sum())
normal_count = total_readings - flagged_count
anomaly_rate = flagged_count / total_readings if total_readings else 0.0

# Experimental indicator only; not a confirmed mechanical-health score.
health_score = max(0, min(100, round(100 * (1 - anomaly_rate))))

st.divider()
st.subheader("Vehicle Analysis Overview")

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Readings Analysed", f"{total_readings:,}")
metric2.metric("Flagged Readings", f"{flagged_count:,}")
metric3.metric("Anomaly Rate", f"{anomaly_rate:.2%}")
metric4.metric("Experimental Health Score", f"{health_score}/100")

st.warning(
    "The health score is an experimental indicator based on the anomaly rate. "
    "It is not a verified diagnosis of mechanical condition."
)


# ============================================================
# 6. SENSOR SELECTION AND READING RANGE
# ============================================================

st.divider()
st.subheader("📈 Sensor Trends")
st.write("Choose a sensor and reading range to inspect the sensor pattern.")

default_sensor_index = (
    feature_names.index("ENGINE_RPM")
    if "ENGINE_RPM" in feature_names
    else 0
)
selected_sensor = st.selectbox(
    "Choose a sensor",
    options=feature_names,
    index=default_sensor_index,
)

max_reading = int(results["Reading Number"].max())
default_start = max(1, max_reading - 999)

start_reading, end_reading = st.slider(
    "Select the reading range",
    min_value=1,
    max_value=max_reading,
    value=(default_start, max_reading),
)

chart_data = results[
    results["Reading Number"].between(start_reading, end_reading)
].copy()

if chart_data.empty:
    st.info("No readings are available in this range.")
else:
    # Downsample normal readings for responsiveness, but retain all flagged
    # points in the selected range.
    if len(chart_data) > 3000:
        normal_data = chart_data.loc[~chart_data["Anomaly"]]
        if len(normal_data) > 2000:
            sample_positions = np.linspace(
                0, len(normal_data) - 1, 2000, dtype=int
            )
            normal_data = normal_data.iloc[sample_positions]

        anomaly_data = chart_data.loc[chart_data["Anomaly"]]
        plotted = (
            pd.concat([normal_data, anomaly_data])
            .drop_duplicates(subset=["Reading Number"])
            .sort_values("Reading Number")
        )
    else:
        plotted = chart_data

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plotted["Reading Number"],
            y=plotted[selected_sensor],
            mode="lines",
            name="Sensor reading",
            line={"width": 1.5},
            connectgaps=False,
            hovertemplate=(
                "Reading number: %{x}<br>"
                + selected_sensor
                + ": %{y}<extra></extra>"
            ),
        )
    )

    flagged_points = plotted.loc[plotted["Anomaly"]]
    if not flagged_points.empty:
        fig.add_trace(
            go.Scatter(
                x=flagged_points["Reading Number"],
                y=flagged_points[selected_sensor],
                mode="markers",
                name="Flagged by model",
                marker={"size": 7, "symbol": "circle"},
                customdata=flagged_points[["Anomaly Error"]].to_numpy(),
                hovertemplate=(
                    "Reading number: %{x}<br>"
                    + selected_sensor
                    + ": %{y}<br>"
                    + "Prediction error: %{customdata[0]:.5f}"
                    + "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"{selected_sensor} Sensor Trend",
        xaxis_title="Sequential Reading Number",
        yaxis_title=selected_sensor,
        template="plotly_dark",
        height=500,
        hovermode="closest",
        legend={"orientation": "h", "yanchor": "bottom",
                "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 20, "r": 20, "t": 80, "b": 30},
    )
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    range_flagged = int(chart_data["Anomaly"].sum())
    st.metric("Flagged Readings in Selected Range", f"{range_flagged:,}")
    st.caption(
        "Markers identify readings flagged by model prediction error. "
        "They are candidates for investigation, not confirmed vehicle faults."
    )


# ============================================================
# 7. ANOMALY SUMMARY AND RESULTS TABLE
# ============================================================

st.divider()
st.subheader("🔎 Anomaly Detection")

if flagged_count:
    st.warning(
        f"The model flagged {flagged_count:,} readings for further inspection."
    )
else:
    st.success("No readings exceeded the current anomaly threshold.")

left, right = st.columns(2)
left.metric("Readings Not Flagged", f"{normal_count:,}")
right.metric("Readings Flagged", f"{flagged_count:,}")

st.divider()
st.subheader("📋 Sensor Reading Details")

show_anomalies_only = st.checkbox("Show only flagged readings")

if show_anomalies_only:
    table_data = results.loc[results["Anomaly"]].copy()
else:
    table_data = results.tail(min(100, len(results))).copy()

preferred_columns = [
    "Reading Number",
    "Timestamp",
    "ENGINE_RPM",
    "ENGINE_COOLANT_TEMP",
    "ENGINE_LOAD",
    "SPEED",
    "FUEL_LEVEL",
    "THROTTLE_POS",
    "Anomaly Error",
    "Anomaly",
]
visible_columns = [col for col in preferred_columns if col in table_data.columns]
visible_columns += [
    col for col in table_data.columns if col not in visible_columns
]
table_data = table_data[visible_columns]

st.dataframe(table_data, use_container_width=True, height=400)


# ============================================================
# 8. DOWNLOAD REPORT
# ============================================================

st.divider()
st.subheader("📥 Download Analysis Report")

csv_data = results.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Vehicle Analysis Report (CSV)",
    data=csv_data,
    file_name="vehicle_health_analysis.csv",
    mime="text/csv",
)

st.caption(f"Dataset: {dataset_name} | Model: LSTM")
st.caption(
    "This application performs experimental OBD-II sensor analysis. "
    "Anomaly flags do not independently establish that a vehicle component is faulty."
)
