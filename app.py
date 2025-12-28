import streamlit as st
import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import joblib

st.set_page_config(layout="wide")
st.title("Anomaly Detection for Physiological Time Series Data")

st.markdown("""
### How the model detects anomalies
The app uses a machine learning model called **Isolation Forest** to find unusual patterns in the signal.

1. The signal is divided into small overlapping **windows**.
2. Each window is compared to the others to see how **different** it is.
3. Windows that are very different from the majority are labeled as **anomalies**.
4. Anomalies are highlighted in **red** on the plots.
""")

# -----------------------------
# Helper functions
# -----------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if high >= 1: high = 0.99
    if low <= 0: low = 0.01
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=125, order=4):
    if fs <= 2*highcut: return signal
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, signal)

def sliding_windows(signal, window_size, step):
    return np.array([signal[i:i+window_size] for i in range(0, len(signal)-window_size+1, step)])

def load_patient_signal(dat_path):
    record_name = os.path.splitext(dat_path)[0]
    record = wfdb.rdrecord(record_name)
    signal = record.p_signal[:, 0]
    fs = record.fs
    return signal, fs

def map_anomalies(signal, windows, model, window_step):
    preds = model.predict(windows)
    anomalies_idx = []
    for i, p in enumerate(preds):
        if p == -1:
            start = i * window_step
            anomalies_idx.extend(range(start, start + len(windows[i])))
    anomalies_idx = np.array(anomalies_idx, dtype=int)
    anomalies_idx = anomalies_idx[anomalies_idx < len(signal)]
    return anomalies_idx

# -----------------------------
# Load pre-trained IsolationForest model
# -----------------------------
model_path = "saved_model/isolation_forest_model.pkl"
model = joblib.load(model_path)
st.success("Loaded pre-trained Isolation Forest model!")

# -----------------------------
# Sample patients
# -----------------------------
sample_folder = "C:\\Users\\NEHA\\Machine_Learning_Projects\\Anomaly detection in Time series data\\sample_patients\\"
sample_files = [f for f in os.listdir(sample_folder) if f.endswith('.dat')]
window_size = 100
window_step = 50

st.subheader("Sample Patients (instant view)")

for dat_file in sample_files:
    dat_path = os.path.join(sample_folder, dat_file)
    patient_name = os.path.splitext(dat_file)[0]
    
    signal, fs = load_patient_signal(dat_path)
    signal = np.nan_to_num(signal)
    signal = bandpass_filter(signal, fs=fs)
    signal_norm = (signal - np.mean(signal)) / np.std(signal)
    
    # Sliding windows & anomalies
    windows = sliding_windows(signal_norm, window_size, window_step)
    anomalies_idx = map_anomalies(signal_norm, windows, model, window_step)
    num_anomalies = len(anomalies_idx)
    
    # Two-column display
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{patient_name} - Original Signal**")
        fig1, ax1 = plt.subplots(figsize=(10,3))
        ax1.plot(signal_norm, color='blue')
        ax1.set_title("Original Signal")
        st.pyplot(fig1)
        st.write(f"**Number of anomalies detected:** N/A")  # Original signal
    
    with col2:
        st.write(f"**{patient_name} - Anomalies Highlighted**")
        fig2, ax2 = plt.subplots(figsize=(10,3))
        ax2.plot(signal_norm, color='blue')
        ax2.scatter(anomalies_idx, signal_norm[anomalies_idx], color='red', label='Anomalies')
        ax2.set_title("Anomalies Highlighted")
        ax2.legend()
        st.pyplot(fig2)
        st.write(f"**Number of anomalies detected:** {num_anomalies}")

# -----------------------------
# Optional: Upload custom patient files
# -----------------------------
st.subheader("Upload Your Own Patient Files (optional)")

uploaded_files = st.file_uploader(
    "Upload .dat and .hea files", type=['dat','hea'], accept_multiple_files=True
)

if uploaded_files:
    temp_dir = "temp_uploaded"
    os.makedirs(temp_dir, exist_ok=True)

    # Save uploaded files
    for f in uploaded_files:
        with open(os.path.join(temp_dir, f.name), "wb") as out:
            out.write(f.getbuffer())
    
    uploaded_dat_files = [f for f in os.listdir(temp_dir) if f.endswith(".dat")]
    
    for dat_file in uploaded_dat_files:
        dat_path = os.path.join(temp_dir, dat_file)
        patient_name = os.path.splitext(dat_file)[0]
        try:
            signal, fs = load_patient_signal(dat_path)
            signal = np.nan_to_num(signal)
            signal = bandpass_filter(signal, fs=fs)
            signal_norm = (signal - np.mean(signal)) / np.std(signal)
            
            windows = sliding_windows(signal_norm, window_size, window_step)
            anomalies_idx = map_anomalies(signal_norm, windows, model, window_step)
            num_anomalies = len(anomalies_idx)
            
            st.write(f"**Uploaded Patient: {patient_name}**")
            fig, ax = plt.subplots(figsize=(12,4))
            ax.plot(signal_norm, label='Signal')
            ax.scatter(anomalies_idx, signal_norm[anomalies_idx], color='red', label='Anomalies')
            ax.set_title(f"Anomalies - {patient_name}")
            ax.legend()
            st.pyplot(fig)
            st.write(f"**Number of anomalies detected:** {num_anomalies}")
            
        except FileNotFoundError:
            st.error(f"Missing .hea file for {dat_file}, skipping.")
