# Generic Anomaly Data Quality (GADQ)

A Python-based data quality and anomaly detection framework for Databricks.

## Overview

This project provides automated data profiling and anomaly detection capabilities for Databricks Delta tables. It is designed to:

1. Profile key fields in tables on a scheduled basis
2. Store profiling results in Delta tables
3. Detect anomalies in the data using statistical and machine learning models
4. Support time-grain based analysis (e.g., business_dt)

## Project Structure

```
.
├── config/                 # Configuration files
├── src/                    # Source code
│   ├── profiling/         # Data profiling module
│   ├── anomaly/           # Anomaly detection module
│   ├── delta/             # Delta table management
│   └── utils/             # Utility functions
├── tests/                 # Test files
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your Databricks connection in `config/config.yaml`

3. Set up your tables to monitor in `config/tables.yaml`

## Usage

Run the main orchestration script:
```python
python src/main.py
```

## Features

- Automated data profiling
- Statistical anomaly detection
- Machine learning-based anomaly detection
- Delta table integration
- Time-grain based analysis
- Configurable profiling metrics
- Historical trend analysis

## Configuration

Edit the following configuration files:
- `config/config.yaml`: Databricks connection and general settings
- `config/tables.yaml`: Tables to monitor and their configurations 