# IKIGAI - Ather Ignition 1.0 🚀 | IoT Smart Helmet

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Event](https://img.shields.io/badge/Hackathon-Ather_Ignition_1.0-ea5d24.svg)
![Domain](https://img.shields.io/badge/Domain-IoT_%7C_Rider_Safety-2ea44f.svg)

Welcome to **IKIGAI**, an IoT Smart Helmet prototype developed for the **Ather Ignition 1.0** hackathon. This repository contains the Python-based backend pipeline designed to ingest real-time hardware sensor data from the helmet, classify rider events, and ensure enhanced rider safety.

## 💡 Project Overview
The IKIGAI smart helmet bridges the gap between the rider and the vehicle. By continuously monitoring hardware sensors embedded in the helmet, this pipeline analyzes the rider's state and detects critical events (e.g., impact detection, helmet usage verification, or behavioral classification) in real-time.

## 📁 System Architecture

The software is structured into three core Python modules that handle the flow from raw IoT sensor data to classified safety alerts:

* **`listener.py` (Data Ingestion)** The data acquisition engine. This script actively listens to the incoming telemetry streams from the helmet's IoT sensors (such as accelerometers, gyroscopes, or microcontrollers like ESP32/Arduino via serial/Bluetooth). It handles raw data formatting and buffering.
  
* **`console_classification.py` (Event Classification)** The brain of the system. It processes the cleaned sensor data passed from the listener and applies classification logic. Whether using threshold-based rules or machine learning models, it identifies specific states like crash impacts, sudden braking, or simply confirming the helmet is being worn correctly, outputting these alerts to the console.
  
* **`main_app.py` (Application Orchestrator)** The central loop. It initializes the connection to the helmet, ties the `listener.py` feed into the `console_classification.py` engine, and maintains the continuous real-time monitoring required for rider safety.


## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone 
   cd IKIGAIpython -m venv venv
2. ** Install Dependencies: **
  ```bash
   source venv/bin/activate  # On Windows use: venv\Scripts\activate

