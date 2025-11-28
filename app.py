import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.title("📊 Customer Segmentation: RFM & K-Means")
st.write("Aplikasi ini mengotomatisasi segmentasi pelanggan menggunakan metode RFM dan K-Means Clustering berdasarkan data transaksi.")

# --- 1. Upload Data ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload file CSV (UrbanMart_Transactions.csv)", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.write("### Preview Data Awal")
    st.dataframe(df.head())

    # --- 2. Data Cleaning (Sesuai Notebook) ---
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Convert date
    # Pastikan nama kolom sesuai dengan notebook Anda: 'TransactionDate'
    if 'TransactionDate' in df.columns:
        df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    
    # Drop missing values
    df = df.dropna(subset=["CustomerID", "TransactionDate"])
    
    # Ensure numeric transaction value
    if 'TransactionValue' in df.columns:
        df["TransactionValue"] = pd.to_numeric(df["TransactionValue"], errors="coerce")
        df = df.dropna(subset=["TransactionValue"])
        # Rename untuk standarisasi jika perlu, tapi di notebook Anda pakai TransactionValue
        # df.rename(columns={"TransactionValue": "TotalAmount"}, inplace=True) 
    
    st.success(f"Data berhasil dimuat dan dibersihkan! Total baris: {df.shape[0]}")

    # --- 3. Visualisasi Dasar (EDA) ---
    st.subheader("📈 Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Distribusi Kategori Produk**")
        if 'ProductCategory' in df.columns:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.countplot(data=df, x="ProductCategory", ax=ax, palette="viridis")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
    with col2:
        st.markdown("**Metode Pembayaran**")
        if 'PaymentMethod' in df.columns:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.countplot(data=df, x="PaymentMethod", ax=ax, palette="pastel")
            st.pyplot(fig)

    # --- 4. Kalkulasi RFM ---
    st.subheader("🧮 RFM Calculation")
    
    # Snapshot date: max date + 1 day
    snapshot_date = df['TransactionDate'].max() + pd.Timedelta(days=1)
    
    # Agregasi RFM sesuai notebook
    rfm = df.groupby('CustomerID').agg({
        'TransactionDate': lambda x: (snapshot_date - x.max()).days,
        'TransactionID': 'count',
        'TransactionValue': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # Filter monetary > 0
    rfm = rfm[rfm['Monetary'] > 0]
    
    st.write("Data RFM (Recency, Frequency, Monetary):")
    st.dataframe(rfm.head())
    
    # --- 5. K-Means Clustering ---
    st.sidebar.header("2. Konfigurasi Model")
    num_clusters = st.sidebar.slider("Pilih Jumlah Kluster (k)", min_value=2, max_value=10, value=4)
    
    # Scaling
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    # Modeling
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # --- 6. Hasil Segmentasi ---
    st.subheader(f"🎯 Hasil Segmentasi (k={num_clusters})")
    
    # Summary Stats per Cluster
    cluster_summary = rfm.groupby("Cluster")[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
    st.table(cluster_summary)
    
    # Visualisasi Scatter Plot (menggunakan PCA untuk reduksi dimensi ke 2D)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(rfm_scaled)
    rfm['PCA1'] = pca_result[:, 0]
    rfm['PCA2'] = pca_result[:, 1]
    
    st.markdown("**Visualisasi Kluster (PCA 2D)**")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=rfm, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', s=60, ax=ax)
    plt.title("Customer Segments")
    st.pyplot(fig)
    
    # Download Result
    st.download_button(
        label="Download Hasil Segmentasi (CSV)",
        data=rfm.to_csv(index=False).encode('utf-8'),
        file_name='customer_segments.csv',
        mime='text/csv',
    )

else:
    st.info("Silakan upload file CSV transaksi untuk memulai.")