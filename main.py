    # streamlit_app.py (Corrected File-based architecture)
# - FIX: Uses sys.executable for robust subprocess creation.
# - FIX: Removes file deletion logic from this script; the listener is responsible for creating a fresh log.
# - FIX: Adds a small delay after starting the listener to prevent a race condition.

import os
import subprocess
import sys  # Import sys to find the correct python executable
import time
import math
from collections import deque
import numpy as np
import pandas as pd
import streamlit as st

# --- Configuration ---
DATA_FILENAME = "imu_stream.csv"
LISTENER_SCRIPT_NAME = "listener.py"

# --- Helper Functions (Unchanged) ---
def calc_accel_magnitude(x, y, z):
    return math.sqrt(x ** 2 + y ** 2 + z ** 2)

def dominant_frequency(accel_series, timestamps):
    if len(accel_series) < 10: return 0.0, None
    dt = np.diff(timestamps)
    if np.any(dt <= 1e-6): fs = 50.0
    else: fs = 1.0 / np.median(dt)
    x = np.array(accel_series) - np.mean(accel_series)
    n = len(x)
    fft = np.fft.rfft(x * np.hanning(n)); psd = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    mask = (freqs >= 0.3)
    if not np.any(mask): return 0.0, fs
    freqs_masked, psd_masked = freqs[mask], psd[mask]
    idx = np.argmax(psd_masked)
    return float(freqs_masked[idx]), fs

def peak_jerk(accel_series, timestamps):
    if len(accel_series) < 3: return 0.0
    a = np.array(accel_series); t = np.array(timestamps)
    dt, da = np.diff(t), np.diff(a)
    dt[dt == 0] = 1e-3
    jerk = da / dt
    return float(np.max(np.abs(jerk)))

def accel_pitch_deg(ax, ay, az):
    denom = math.sqrt(max(1e-9, ay*ay + az*az))
    pitch = math.atan2(ax, denom)
    return abs(math.degrees(pitch))

# --- State Initialization ---
def initialize_session_state():
    if 'is_listening' not in st.session_state:
        st.session_state.is_listening = False
    if 'listener_process' not in st.session_state:
        st.session_state.listener_process = None
    if 'current_activity' not in st.session_state:
        st.session_state.current_activity = "idle"
    if 'last_activity_change_time' not in st.session_state:
        st.session_state.last_activity_change_time = time.time()
    if 'features' not in st.session_state:
        st.session_state.features = {}

# --- Activity Classification Logic (Operates on a DataFrame) ---
def classify_activity(df_window, config):
    if df_window.empty:
        return "idle", {}
    
    timestamps = df_window['timestamp'].values
    axs, ays, azs = df_window['accel_x'].values, df_window['accel_y'].values, df_window['accel_z'].values
    mags = [calc_accel_magnitude(x, y, z) for x, y, z in zip(axs, ays, azs)]

    accel_peak = float(np.max(mags)) if mags else 0.0
    dom_freq, fs = dominant_frequency(mags, timestamps)
    jerk_peak = peak_jerk(mags, timestamps)
    pitches = [accel_pitch_deg(x, y, z) for x, y, z in zip(axs, ays, azs)]
    pitch_mean = float(np.mean(pitches)) if pitches else 0.0

    features = {
        'Accel Peak (m/s²)': accel_peak, 'Dominant Freq (Hz)': dom_freq,
        'Peak Jerk (m/s³)': jerk_peak, 'Mean Pitch (°)': pitch_mean, 'Est. Fs (Hz)': fs
    }
    
    a_check = accel_peak
    if a_check <= config['IDLE_MAX']: candidate = "idle"
    elif config['WALK_MIN'] < a_check <= config['WALK_MAX']: candidate = "walking"
    elif config['RUN_MIN'] < a_check <= config['RUN_MAX']: candidate = "scooter"
    elif config['RIDE_MIN'] <= a_check: candidate = "ride_candidate"
    else: candidate = "idle"

    label = candidate
    if candidate == "scooter" and config['PERIODIC_FREQ_MIN'] <= dom_freq <= config['PERIODIC_FREQ_MAX']:
        label = "scooter"
    elif candidate == "ride_candidate":
        if config['PERIODIC_FREQ_MIN'] <= dom_freq <= config['PERIODIC_FREQ_MAX'] and a_check > config['RUN_MIN']:
            label = "scooter"
        elif jerk_peak >= config['JERK_THRESHOLD']:
            label = "bike" if pitch_mean >= config['PITCH_BIKE_DEG'] else "bike"
        else:
            if a_check >= (config['RIDE_MIN'] + config['RUN_MAX']) / 2.0:
                label = "bike" if pitch_mean >= config['PITCH_BIKE_DEG'] else "bike"
            else:
                label = "scooter"
    
    now = time.time()
    if label != st.session_state.current_activity:
        if now - st.session_state.last_activity_change_time >= config['MIN_DURATION']:
            st.session_state.current_activity = label
            st.session_state.last_activity_change_time = now
    else:
        st.session_state.last_activity_change_time = now
        
    st.session_state.features = features

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Real-Time IMU Activity Classifier")
st.title("🏃 Real-Time IMU Activity Classifier")

initialize_session_state()

with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("Listener Control")
    col1, col2 = st.columns(2)
    if col1.button("Start Listener", type="primary", disabled=st.session_state.is_listening):
        # FIX: Use sys.executable to ensure we use the same python interpreter.
        # This is more robust than just "python".
        python_executable = sys.executable
        
        # Start the listener.py script as a background process.
        # The listener script is responsible for creating/clearing the log file.
        st.session_state.listener_process = subprocess.Popen([python_executable, LISTENER_SCRIPT_NAME])
        st.session_state.is_listening = True
        
        # FIX: Give the listener a moment to start and create the file
        # before the main app loop tries to read it.
        time.sleep(0.5)
        st.rerun()

    if col2.button("Stop Listener", disabled=not st.session_state.is_listening):
        if st.session_state.listener_process:
            st.session_state.listener_process.terminate() # Send signal to stop
            st.session_state.listener_process.wait() # Wait for it to close
            st.session_state.listener_process = None
        st.session_state.is_listening = False
        st.rerun()

    st.subheader("Acceleration Ranges (m/s²)")
    idle_max = st.slider("Idle Max Accel", 0.0, 1.0, 0.1, 0.01)
    walk_min, walk_max = st.slider("Walk Accel Range", 0.0, 20.0, (0.1, 15.0))
    run_min, run_max = st.slider("Run Accel Range", 0.0, 40.0, (5.0, 30.0))
    ride_min, ride_max = st.slider("Ride Accel Range", 0.0, 60.0, (5.0, 50.0))
    st.subheader("Detection Parameters")
    window_sec = st.slider("Analysis Window (sec)", 0.5, 5.0, 2.0, 0.1)
    jerk_threshold = st.slider("Jerk Threshold (m/s³)", 1.0, 50.0, 10.0, 0.5)
    periodic_freq_min, periodic_freq_max = st.slider("Periodic Freq Range (Hz)", 0.1, 10.0, (0.7, 4.0))
    pitch_bike_deg = st.slider("Bike Pitch Threshold (°)", 0.0, 90.0, 40.0, 1.0)
    min_duration = st.slider("Min Stable Duration (sec)", 0.1, 5.0, 1.0, 0.1)

    config = {
        "IDLE_MAX": idle_max, "WALK_MIN": walk_min, "WALK_MAX": walk_max, "RUN_MIN": run_min,
        "RUN_MAX": run_max, "RIDE_MIN": ride_min, "RIDE_MAX": ride_max, "WINDOW_SEC": window_sec,
        "JERK_THRESHOLD": jerk_threshold, "PERIODIC_FREQ_MIN": periodic_freq_min,
        "PERIODIC_FREQ_MAX": periodic_freq_max, "PITCH_BIKE_DEG": pitch_bike_deg,
        "MIN_DURATION": min_duration
    }
    
# --- Main Dashboard ---
if st.session_state.is_listening:
    st.info(f"🟢 Listener process is active. Reading data from '{DATA_FILENAME}'.")
else:
    st.warning("🔴 Listener is stopped. Press 'Start Listener' to begin.")

col1, col2 = st.columns([1, 2])
current_activity_display = col1.empty()
features_display = col2.empty()

st.markdown("---")
st.header("Live Sensor Data")
accel_chart_placeholder = st.empty()
gyro_chart_placeholder = st.empty()

while True:
    df_plot = pd.DataFrame() # Default to empty dataframe
    if st.session_state.is_listening and os.path.exists(DATA_FILENAME):
        try:
            # Read the entire file
            df = pd.read_csv(DATA_FILENAME)
            if not df.empty:
                # Get the latest window of data for classification
                latest_timestamp = df['timestamp'].iloc[-1]
                window_start_time = latest_timestamp - config['WINDOW_SEC']
                df_window = df[df['timestamp'] >= window_start_time]
                
                classify_activity(df_window, config)
                
                # Prepare data for plotting (last 500 points for performance)
                df_plot = df.tail(500).copy()
                df_plot['datetime'] = pd.to_datetime(df_plot['timestamp'], unit='s')
                df_plot.set_index('datetime', inplace=True)
                
        except pd.errors.EmptyDataError:
            # This can happen if the file is created but not yet written to.
            st.toast("Log file is empty, waiting for data...")
        except Exception as e:
            st.error(f"Error reading or processing data file: {e}")

    # Update UI elements
    current_activity_display.metric(label="Current Activity", value=st.session_state.current_activity.upper())
    with features_display:
        st.write("**Classification Features (from analysis window):**")
        if st.session_state.features:
            formatted_features = {k: f"{v:.2f}" for k, v in st.session_state.features.items() if v is not None}
            st.json(formatted_features)
        else:
            st.info("Waiting for data...")

    with accel_chart_placeholder.container():
        st.subheader("Accelerometer (m/s²)")
        if not df_plot.empty:
            st.line_chart(df_plot[['accel_x', 'accel_y', 'accel_z']])
        else:
            st.info("Waiting for accelerometer data...")
            
    with gyro_chart_placeholder.container():
        st.subheader("Gyroscope (rad/s)")
        if not df_plot.empty:
            st.line_chart(df_plot[['gyro_x', 'gyro_y', 'gyro_z']], color=["#00BFFF", "#FF00FF", "#FFA500"])
        else:
            st.info("Waiting for gyroscope data...")

    time.sleep(0.2)
