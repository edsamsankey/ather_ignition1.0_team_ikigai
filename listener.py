# listener.py
# A standalone script to listen for UDP packets and write them to a CSV file.
# This script should NOT be run directly. The Streamlit app will manage it.

import socket
import re
import time
import csv
import logging
from datetime import datetime

# --- Configuration ---
PORT = 2055
BUFFER_SIZE = 8192
OUTPUT_FILENAME = "imu_stream.csv"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def start_udp_listener():
    """
    Listens for UDP packets and appends them to a CSV file.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', PORT))
        logging.info(f"UDP Listener starting on port {PORT}. Writing to {OUTPUT_FILENAME}")
    except Exception as e:
        logging.error(f"FATAL: Error binding socket: {e}")
        return

    # Open the file once in append mode
    with open(OUTPUT_FILENAME, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])
        
        while True:
            try:
                data, _ = s.recvfrom(BUFFER_SIZE)
                packet_ts = time.time() # Use high-precision timestamp
                
                data_str = data.decode('utf-8').strip()
                clean_str = re.sub(r'[\r\n\t]+', '', data_str)
                clean_str = re.sub(r',+', ',', clean_str).strip(',')
                
                vals = [float(v) if v else 0 for v in clean_str.split(',')[:6]]
                while len(vals) < 6: vals.append(0.0)
                
                # Write the timestamp and the 6 sensor values
                writer.writerow([packet_ts, *vals])
                file.flush() # Ensure data is written immediately

            except Exception as e:
                logging.warning(f"Error processing packet: {e}")
                
    s.close()
    logging.info("UDP Listener shut down.")


if __name__ == "__main__":
    start_udp_listener()
