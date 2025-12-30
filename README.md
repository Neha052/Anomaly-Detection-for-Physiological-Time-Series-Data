# Anomaly-Detection-for-Physiological-Time-Series-Data
Technologies Used
![Static Badge](https://img.shields.io/badge/Python-blue)  ![Static Badge](https://img.shields.io/badge/scikitlearn-orange)  ![Static Badge](https://img.shields.io/badge/Streamlit-App-red) ![Static Badge](https://img.shields.io/badge/pandas-yello) ![Static Badge](https://img.shields.io/badge/numpy-yello) ![Static Badge](https://img.shields.io/badge/matlab-yellow) ![StaticBadge](https://img.shields.io/badge/pickle-yellow) ![Static Badge](https://img.shields.io/badge/scipy-yellow) ![Static Badge](https://img.shields.io/badge/WFDB-red)


🔗 Live App:
👉 [https://anomaly-detection-for-physiological-time-series-data.streamlit.app/)]

## How to Use the App:

### 1. View Sample Results (Instant)

  * Sample patient signals load automatically.

  * For each patient:

    * Left: Original signal

    * Right: Signal with detected anomalies highlighted.

* Displays number of anomalies detected per patient.

### 2. Upload Your Own Data (Optional)

  * Upload a matching .dat and .hea file.

  * Signals are processed and visualized immediately.

## Problem Statement 

Physiological time-series data is:

* High-frequency and noisy.

* Difficult to inspect manually.

* Often unlabeled, making supervised learning impractical.

The objective is to **identify unusual signal segments (e.g., abnormal peaks)** that warrant further inspection, without relying on labeled anomalies.

## Modeling Approach

* Unsupervised anomaly detection using **Isolation Forest**.

* Sliding-window representation to capture local temporal patterns.

* Model trained across multiple patients to learn population-level normal behavior.

* Anomalous windows mapped back to signal indices for interpretability.

This mirrors real-world data science workflows where labels are unavailable and models support downstream decision-making.

## How Anomalies Are Detected:

* Each signal is split into overlapping windows.

* The model learns what “normal” windows look like.

* Windows that differ significantly from most others are flagged.

* Flagged regions typically correspond to unusual peaks or irregular patterns.

* These regions are highlighted in the visualization.

## Data Science Value

* Reduces long time-series into small, high-interest segments.

* Enables scalable analysis across patients.

* Improves consistency compared to manual inspection.

* Acts as a first-pass screening layer, not a diagnostic system.

## What This Project Demonstrates

* Time-series preprocessing and windowing.

* Unsupervised ML on real-world data.

* Model persistence and reuse.

* Translating ML output into interpretable visualizations.

* End-to-end deployment beyond notebooks.

## Limitations & Future Work

1. No ground-truth labels (by design).

2. Anomalies indicate unusual patterns, not clinical events.

## Future work:

* Patient-level anomaly summaries.

* Model comparison (LOF, autoencoders).

* Interactive parameter tuning.
  
## Author

Neha Ingale
