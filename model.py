from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "cleaned"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

WINDOW_SIZE = 10
EPOCHS = 5


def load_data():
    files = sorted(DATA_DIR.glob("*_cleaned.csv"))

    if not files:
        raise FileNotFoundError(
            "No cleaned CSV files found. Run prepare_data.py first."
        )

    frames = []

    for file in files:
        df = pd.read_csv(file, low_memory=False)

        # Keep rows with valid timestamps and sort chronologically.
        if "TIMESTAMP" in df.columns:
            df["TIMESTAMP"] = pd.to_datetime(
                df["TIMESTAMP"], errors="coerce"
            )
            df = df.dropna(subset=["TIMESTAMP"])
            df = df.sort_values("TIMESTAMP")

        frames.append(df)

    # Use one experiment file at a time because the files may represent
    # different vehicles and driving conditions.
    return frames[0]


def prepare_sensor_data(df):
    excluded = {
        "TIMESTAMP",
        "TROUBLE_CODES",
        "DTC_NUMBER",
        "MARK",
        "MODEL",
        "VEHICLE_ID",
        "FUEL_TYPE",
        "AUTOMATIC",
    }

    sensor_columns = []

    for col in df.columns:
        if col in excluded:
            continue

        values = pd.to_numeric(df[col], errors="coerce")

        if values.notna().sum() > 0:
            df[col] = values
            sensor_columns.append(col)

    if not sensor_columns:
        raise ValueError("No numeric sensor columns were found.")

    sensors = df[sensor_columns].replace([np.inf, -np.inf], np.nan)

    # Remove entirely empty columns and fill remaining gaps.
    sensors = sensors.dropna(axis=1, how="all")
    sensors = sensors.interpolate(limit_direction="both").ffill().bfill()
    sensors = sensors.dropna()

    if len(sensors) <= WINDOW_SIZE:
        raise ValueError("Not enough valid readings to train the LSTM.")

    return sensors


def create_sequences(values, window_size):
    X, y = [], []

    for i in range(window_size, len(values)):
        X.append(values[i - window_size:i])
        y.append(values[i])

    return np.asarray(X), np.asarray(y)


def train_model():
    df = load_data()
    sensors = prepare_sensor_data(df)

    # Split chronologically to avoid training on future readings.
    split = int(len(sensors) * 0.8)

    if split <= WINDOW_SIZE or len(sensors) - split < 1:
        raise ValueError("Not enough readings for a training/test split.")

    scaler = MinMaxScaler()
    scaler.fit(sensors.iloc[:split])

    scaled = scaler.transform(sensors)

    X, y = create_sequences(scaled, WINDOW_SIZE)

    # The target timestamp is at index i, so preserve chronological split.
    target_indices = np.arange(WINDOW_SIZE, len(scaled))
    train_mask = target_indices < split
    test_mask = target_indices >= split

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    model = Sequential([
        Input(shape=(WINDOW_SIZE, X.shape[2])),
        LSTM(32),
        Dense(X.shape[2]),
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=32,
        validation_split=0.1 if len(X_train) >= 20 else 0,
        verbose=1,
    )

    predictions = model.predict(X_test, verbose=0)
    errors = np.mean(np.square(y_test - predictions), axis=1)

    # Initial threshold: 95th percentile of test prediction errors.
    # This is a provisional anomaly threshold, not a validated fault diagnosis.
    threshold = float(np.percentile(errors, 95))

    model.save(MODEL_DIR / "vehicle_lstm.keras")

    np.savez(
        MODEL_DIR / "scaler_data.npz",
        scale=scaler.scale_,
        min=scaler.min_,
        data_min=scaler.data_min_,
        data_max=scaler.data_max_,
        feature_names=np.asarray(sensors.columns, dtype=str),
        threshold=threshold,
        window_size=WINDOW_SIZE,
    )

    results = pd.DataFrame({
        "actual_error": errors,
        "is_anomaly": errors > threshold,
    })

    results.to_csv(MODEL_DIR / "test_anomalies.csv", index=False)

    print("\nTraining completed!")
    print("Sensor columns:", list(sensors.columns))
    print("Test readings:", len(errors))
    print("Provisional anomaly threshold:", threshold)
    print("Model saved to:", MODEL_DIR / "vehicle_lstm.keras")


if __name__ == "__main__":
    train_model()