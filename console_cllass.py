# realtime_imu_activity_with_ride_tilt.py
# Edited from user's original per request:
# - remove "running" as a label
# - treat any "running" candidate as a ride candidate (scooter/bike logic)
# - decide scooter vs bike using tilt (pitch) threshold
# - keeps CSV logging and plotting

import logging
import csv
import os
import socket
import sys
import re
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import math
from datetime import datetime
from collections import deque
import numpy as np

# --- Configuration (tune as needed) ---
PROTOCOL = "UDP"
PORT = 2055
FILENAME = "wireless_imu_data_classified.csv"
BUFFER_SIZE = 8192
MAX_POINTS = 500

# Acceleration ranges (m/s^2)
IDLE_MAX = 0.1
WALK_MIN, WALK_MAX = 0.1, 15.0
# RUN range is removed — values that previously fell in RUN will be handled as ride-candidate
RUN_MIN, RUN_MAX = 5.0, 30.0
RIDE_MIN, RIDE_MAX = 5.0, 50.0

# Windowing / detection params
WINDOW_SEC = 2.0         # seconds of recent data to examine
FS_EST = 50.0            # estimated sampling rate (used for freq axis only)
JERK_THRESHOLD = 50.0     # m/s^3 (threshold for sudden acceleration -> ride)
PERIODIC_FREQ_MIN = 0.7  # Hz ; lower bound for human step periodicity (still used to avoid misclassifying steps)
PERIODIC_FREQ_MAX = 4.0  # Hz ; upper bound for human step periodicity
PITCH_BIKE_DEG = 25.0    # degrees threshold to decide bike vs scooter

# Stability / debouncing
MIN_DURATION = 1.0       # sec consistent before accepting new activity

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)

# --- Global data storage ---
plot_data = {
    'time': [],
    'accel_x': [], 'accel_y': [], 'accel_z': [],
    'gyro_x': [], 'gyro_y': [], 'gyro_z': [],
}
data_lock = threading.Lock()

# rolling buffer for recent accel magnitude and timestamps (for detection)
recent_buf = deque()  # stores tuples (ts, ax, ay, az)

# State tracking
current_activity = "idle"
last_activity_change_time = time.time()
activity_start_time = None
activity_log = []

# --- Helper: compute acceleration magnitude ---
def calc_accel_magnitude(x, y, z):
    return math.sqrt(x ** 2 + y ** 2 + z ** 2)

# --- Helper: dominant frequency in window via FFT (returns freq in Hz) ---
def dominant_frequency(accel_series, timestamps):
    if len(accel_series) < 6:
        return 0.0, None
    dt = np.diff(timestamps)
    if np.any(dt <= 0):
        fs = FS_EST
    else:
        fs = 1.0 / np.median(dt)
    x = np.array(accel_series) - np.mean(accel_series)
    n = len(x)
    fft = np.fft.rfft(x * np.hanning(n))
    psd = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    mask = (freqs >= 0.3) & (freqs <= fs/2.0)
    if not np.any(mask):
        return 0.0, fs
    freqs_masked = freqs[mask]
    psd_masked = psd[mask]
    idx = np.argmax(psd_masked)
    return float(freqs_masked[idx]), fs

# --- Helper: compute peak jerk in window ---
def peak_jerk(accel_series, timestamps):
    if len(accel_series) < 3:
        return 0.0
    a = np.array(accel_series)
    t = np.array(timestamps)
    dt = np.diff(t)
    da = np.diff(a)
    dt[dt == 0] = np.median(dt[dt != 0]) if np.any(dt != 0) else 1.0/FS_EST
    jerk = da / dt
    return float(np.max(np.abs(jerk)))

# --- compute pitch (approx) from accel vector ---
def accel_pitch_deg(ax, ay, az):
    denom = math.sqrt(max(1e-9, ay*ay + az*az))
    pitch = math.atan2(ax, denom)
    return abs(math.degrees(pitch))

# --- Activity classification using the recent window + heuristics ---
def classify_activity_window():
    """
    Uses recent_buf to decide activity.
    Returns (label, features_dict)
    features: accel_mean, accel_peak, dominant_freq, jerk_peak, pitch_deg_mean
    """
    if not recent_buf:
        return "idle", {}

    data = list(recent_buf)
    timestamps = [d[0] for d in data]
    axs = [d[1] for d in data]
    ays = [d[2] for d in data]
    azs = [d[3] for d in data]
    mags = [calc_accel_magnitude(x,y,z) for x,y,z in zip(axs, ays, azs)]

    accel_mean = float(np.mean(mags))
    accel_peak = float(np.max(mags))
    dom_freq, fs = dominant_frequency(mags, timestamps)
    jerk_peak = float(peak_jerk(mags, timestamps))
    pitches = [accel_pitch_deg(x,y,z) for x,y,z in zip(axs, ays, azs)]
    pitch_mean = float(np.mean(pitches))

    features = {
        'accel_mean': accel_mean,
        'accel_peak': accel_peak,
        'dominant_freq': dom_freq,
        'jerk_peak': jerk_peak,
        'pitch_mean': pitch_mean,
        'fs_est': fs
    }

    # Basic candidate from accel_peak
    a_check = accel_peak
    if abs(a_check) <= IDLE_MAX:
        candidate = "idle"
    elif WALK_MIN < a_check <= WALK_MAX:
        candidate = "walking"
    elif RIDE_MIN <= a_check <= RIDE_MAX or a_check > RIDE_MAX:
        candidate = "ride_candidate"
    else:
        # values that previously mapped to run are treated as ride_candidate now
        candidate = "ride_candidate"

    # Final labeling:
    label = candidate

    if candidate == "ride_candidate":
        # If periodic human stepping is strong, it might be human movement -> treat as scooter path (no running)
        if PERIODIC_FREQ_MIN <= dom_freq <= PERIODIC_FREQ_MAX and accel_peak < RIDE_MIN:
            # periodic but small peak -> walking-like; keep walking
            label = "walking"
        elif jerk_peak >= JERK_THRESHOLD or accel_peak >= RIDE_MIN:
            # strong jerk or clearly large accel -> riding
            if pitch_mean >= PITCH_BIKE_DEG:
                label = "bike"
            else:
                label = "scooter"
        else:
            # fallback: treat as scooter (since running removed)
            if pitch_mean >= PITCH_BIKE_DEG:
                label = "bike"
            else:
                label = "scooter"

    # walking and idle remain as-is
    return label, features

# --- Function to handle real-time activity logic (debounced) ---
def handle_activity_classification(accel_x, accel_y, accel_z, ts):
    global current_activity, last_activity_change_time, activity_start_time

    recent_buf.append((ts, accel_x, accel_y, accel_z))
    cutoff = ts - WINDOW_SEC
    while recent_buf and recent_buf[0][0] < cutoff:
        recent_buf.popleft()

    label, feats = classify_activity_window()
    now = time.time()

    if label != current_activity:
        if now - last_activity_change_time >= MIN_DURATION:
            timestamp = datetime.now().strftime("%H:%M:%S")
            if activity_start_time:
                end_time = timestamp
                activity_log.append((activity_start_time, end_time, current_activity))
                print(f"🕒 {activity_start_time} → {end_time}: {current_activity}")

            current_activity = label
            activity_start_time = timestamp
            print(f"🔄 Activity changed to: {label} | feats: {feats}")
            last_activity_change_time = now
        else:
            pass
    else:
        if activity_start_time is None:
            activity_start_time = datetime.now().strftime("%H:%M:%S")

# --- Animation plot function ---
def animate(i, lines_accel, axes_accel, lines_gyro, axes_gyro):
    with data_lock:
        if not plot_data['time']:
            return lines_accel + lines_gyro

        current_time = plot_data['time']

        lines_accel[0].set_data(current_time, plot_data['accel_x'])
        lines_accel[1].set_data(current_time, plot_data['accel_y'])
        lines_accel[2].set_data(current_time, plot_data['accel_z'])
        axes_accel.set_xlim(current_time[0], current_time[-1])

        all_y_accel = plot_data['accel_x'] + plot_data['accel_y'] + plot_data['accel_z']
        if all_y_accel:
            axes_accel.set_ylim(min(all_y_accel) * 1.1, max(all_y_accel) * 1.1)

        lines_gyro[0].set_data(current_time, plot_data['gyro_x'])
        lines_gyro[1].set_data(current_time, plot_data['gyro_y'])
        lines_gyro[2].set_data(current_time, plot_data['gyro_z'])
        axes_gyro.set_xlim(current_time[0], current_time[-1])

        all_y_gyro = plot_data['gyro_x'] + plot_data['gyro_y'] + plot_data['gyro_z']
        if all_y_gyro:
            axes_gyro.set_ylim(min(all_y_gyro) * 1.1, max(all_y_gyro) * 1.1)

    return lines_accel + lines_gyro

# --- UDP Listener ---
def start_udp_listener(filename, port):
    HOST = ''
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        s.bind((HOST, port))
    except Exception as e:
        logging.error(f"Error binding socket: {e}")
        return

    logging.info(f"UDP Listener started on port {port}. Waiting for IMU data...")

    current_time = 0.0
    try:
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ", "Activity"])
            while True:
                data, addr = s.recvfrom(BUFFER_SIZE)
                packet_received_ts = time.time()
                data_str = data.decode('utf-8').strip()
                try:
                    clean_str = re.sub(r'[\r\n\t]+', '', data_str)
                    clean_str = re.sub(r',+', ',', clean_str).strip(',')
                    vals = clean_str.split(',')
                    vals = vals + ['0'] * (6 - len(vals))
                    sensor_values = [float(v) if v else 0 for v in vals[:6]]

                    accel_x, accel_y, accel_z = sensor_values[0:3]
                    gyro_x, gyro_y, gyro_z = sensor_values[3:6]

                    handle_activity_classification(accel_x, accel_y, accel_z, packet_received_ts)

                    with data_lock:
                        current_time += 1
                        plot_data['time'].append(current_time)
                        plot_data['accel_x'].append(accel_x)
                        plot_data['accel_y'].append(accel_y)
                        plot_data['accel_z'].append(accel_z)
                        plot_data['gyro_x'].append(gyro_x)
                        plot_data['gyro_y'].append(gyro_y)
                        plot_data['gyro_z'].append(gyro_z)

                        if len(plot_data['time']) > MAX_POINTS:
                            for key in plot_data:
                                plot_data[key].pop(0)

                    ts = datetime.now().strftime("%H:%M:%S")
                    writer.writerow([ts, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, current_activity])
                    file.flush()

                    print(f"[{ts}] Accel: ({accel_x:.2f}, {accel_y:.2f}, {accel_z:.2f}) "
                          f"Gyro: ({gyro_x:.2f}, {gyro_y:.2f}, {gyro_z:.2f}) → {current_activity}")

                except Exception as e:
                    logging.warning(f"Failed to process packet: {e}")
    except KeyboardInterrupt:
        print("\nStopping listener...")
    finally:
        s.close()
        logging.info("UDP Listener shut down.")

# --- Main ---
if __name__ == "__main__":
    print(f"Starting Real-Time IMU Activity Classifier on {PROTOCOL} port {PORT}...")

    listener_thread = threading.Thread(target=start_udp_listener, args=(FILENAME, PORT), daemon=True)
    listener_thread.start()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.canvas.manager.set_window_title('Live IMU Data Stream (Activity Detection)')

    # Accelerometer subplot
    axes[0].set_title('Accelerometer (m/s²)')
    axes[0].set_ylabel('Accel Value')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    lines_accel = [axes[0].plot([], [], label=f'Accel {a}', color=c)[0]
                   for a, c in zip('XYZ', ['red', 'green', 'blue'])]
    axes[0].legend()

    # Gyroscope subplot
    axes[1].set_title('Gyroscope (rad/s)')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Gyro Value')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    lines_gyro = [axes[1].plot([], [], label=f'Gyro {a}', color=c)[0]
                  for a, c in zip('XYZ', ['cyan', 'magenta', 'orange'])]
    axes[1].legend()

    plt.tight_layout()
    ani = animation.FuncAnimation(fig, animate, fargs=(lines_accel, axes[0], lines_gyro, axes[1]),
                                  interval=100, cache_frame_data=False)
    plt.show()
