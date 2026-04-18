# IKIGAI - Ather Ignition 1.0 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Event](https://img.shields.io/badge/Hackathon-Ather_Ignition_1.0-ea5d24.svg)

Welcome to **IKIGAI**, a project developed for the **Ather Ignition 1.0** hackathon. This repository contains a streamlined Python-based pipeline designed for real-time data ingestion and classification. 

## Repository Structure

The project is modularized into three core Python scripts, keeping the data processing, machine learning classification, and execution logic distinct:

* **`listener.py`** The data ingestion engine. This module is responsible for listening to incoming streams—whether that is real-time EV telemetry, sensor data, or external API inputs—and preparing the raw data for processing.
  
* **`console_classification.py`** The core analytical module. It processes the data captured by the listener, applies classification logic, and outputs categorized results or predictions directly to the console. 
  
* **`main.py**` The central application script. It orchestrates the workflow by linking the `listener.py` feed with the `console_classification.py` logic, running the continuous loop required for the system.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone 
   cd IKIGAIpython -m venv venv
2. ** Install Dependencies: **
  ```bash
   source venv/bin/activate  # On Windows use: venv\Scripts\activate

