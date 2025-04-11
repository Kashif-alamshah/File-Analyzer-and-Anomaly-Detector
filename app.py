import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def read_csv(file):
    return pd.read_csv(file)

def read_txt(file):
    return file.getvalue().decode("utf-8").splitlines()

def read_log(file):
    return [line.strip() for line in file.getvalue().decode("utf-8").splitlines() if "ERROR" in line or "WARNING" in line]

def analyze_csv_data(df):
    numeric_df = df.select_dtypes(include=[np.number])
    
    st.subheader("Summary Statistics")
    st.write(numeric_df.describe())
    
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    
    plt.figure(figsize=(8,5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Matrix")
    st.pyplot(plt)

def detect_anomalies(df):
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df = df.select_dtypes(include=[np.number]).dropna()
    if df.shape[1] < 2:
        st.warning("Not enough numeric columns for clustering.")
        return
    
    df["Cluster"] = kmeans.fit_predict(df)
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], hue="Cluster", palette="viridis")
    plt.title("Clustering Anomalies")
    st.pyplot(plt)

st.title("File Analyzer and Anomaly Detector")
uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        file_name = file.name
        st.subheader(f"Processing: {file_name}")
        
        if file_name.endswith(".csv"):
            df = read_csv(file)
            analyze_csv_data(df)
            detect_anomalies(df)
        elif file_name.endswith(".txt"):
            text_data = read_txt(file)
            st.subheader("Text File Content Preview")
            st.write("\n".join(text_data[:5]))
        elif file_name.endswith(".log"):
            log_data = read_log(file)
            st.subheader("Log File Errors & Warnings")
            st.write("\n".join(log_data))
