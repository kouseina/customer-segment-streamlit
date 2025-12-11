import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Import plotting libraries dengan error handling (sesuai kode Anda)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOTTING_AVAILABLE = True
except Exception:
    _PLOTTING_AVAILABLE = False

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.title("Customer Segmentation: RFM & K-Means")
st.write("Aplikasi ini mengotomatisasi segmentasi pelanggan menggunakan metode RFM dan K-Means Clustering.")

# --- 1. LOGIKA LOAD DATA ---
# Nama file default di project Anda
DEFAULT_FILE_PATH = 'urbanmart_data.csv'

df = None
data_source_info = ""

df = pd.read_csv(DEFAULT_FILE_PATH)
data_source_info = f"Menggunakan Data Default Project ({DEFAULT_FILE_PATH})"

# --- JIKA DATA BERHASIL DILOAD ---
if df is not None:
    st.info(f"📂 **Status Data:** {data_source_info}")
    
    with st.expander("Lihat Data Awal"):
        st.dataframe(df.head())

    # --- 2. Data Cleaning ---
    df.drop_duplicates(inplace=True)
    
    if 'TransactionDate' in df.columns:
        df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    
    df = df.dropna(subset=["CustomerID", "TransactionDate"])
    
    if 'TransactionValue' in df.columns:
        df["TransactionValue"] = pd.to_numeric(df["TransactionValue"], errors="coerce")
        df = df.dropna(subset=["TransactionValue"])
    
    st.success(f"✅ Data siap diproses! Total transaksi: {df.shape[0]}")

    # --- 3. Visualisasi Dasar (EDA) ---
    st.divider()
    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Distribusi Kategori Produk**")
        if 'ProductCategory' in df.columns and _PLOTTING_AVAILABLE:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.countplot(data=df, y="ProductCategory", ax=ax, palette="viridis", order=df['ProductCategory'].value_counts().index)
            st.pyplot(fig)
            
    with col2:
        st.markdown("**Metode Pembayaran**")
        if 'PaymentMethod' in df.columns and _PLOTTING_AVAILABLE:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.countplot(data=df, x="PaymentMethod", ax=ax, palette="pastel")
            st.pyplot(fig)

    # --- 4. Kalkulasi RFM ---
    st.divider()
    st.subheader("RFM Calculation")
    
    snapshot_date = df['TransactionDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg({
        'TransactionDate': lambda x: (snapshot_date - x.max()).days,
        'TransactionID': 'count',
        'TransactionValue': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    rfm = rfm[rfm['Monetary'] > 0]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Rata-rata Recency", f"{rfm['Recency'].mean():.0f} Hari")
    c2.metric("Rata-rata Frequency", f"{rfm['Frequency'].mean():.0f} Kali")
    c3.metric("Rata-rata Monetary", f"Rp {rfm['Monetary'].mean():,.0f}")

    # --- 5. K-Means Clustering ---
    # st.sidebar.header("2. Konfigurasi Model")
    # num_clusters = st.sidebar.slider("Jumlah Kluster (k)", 2, 8, 4)
    num_clusters = 4
    
    # Scaling
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    # Modeling
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # --- 6. Hasil Segmentasi ---
    st.subheader(f"Hasil Segmentasi ({num_clusters} Kluster)")
    
    cluster_summary = rfm.groupby("Cluster")[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
    
    # Visualisasi PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(rfm_scaled)
    rfm['PCA1'] = pca_result[:, 0]
    rfm['PCA2'] = pca_result[:, 1]
    
    if _PLOTTING_AVAILABLE:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=rfm, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', s=60, ax=ax)
        plt.title("Visualisasi Kluster Pelanggan")
        st.pyplot(fig)

    # --- 7. Marketing Strategy Recommendation ---
    st.divider()
    st.subheader("Rekomendasi Strategi Pemasaran")

    # Definisi Strategi per Kluster (Sesuai User Request)
    # Definisi 4 Profil Strategi dari User (Template)
    # Kita akan masukkan Cluster ID secara dinamis ke sini nanti
    STRATEGY_TEMPLATES = {
        "best": {
            "label": "High-Value Loyalists (Champions)",
            "description": "Kluster ini adalah pelanggan terbaik, dengan frekuensi paling tinggi dan monetary terbesar.",
            "strategies": [
                "Berikan benefit VIP & akses eksklusif.",
                "Sediakan rewards premium bernilai besar.",
                "Personalisasi promosi berdasarkan pola belanja.",
                "Pertahankan mereka lewat tier loyalitas tertinggi."
            ],
            "color": "green"
        },
        "potential": {
            "label": "High-Value Loyalists (Champions)",
            "description": "Pelanggan di kluster ini baru bertransaksi, cukup sering berbelanja, dan memiliki nilai belanja tinggi.",
            "strategies": [
                "Berikan akses khusus & layanan prioritas.",
                "Tawarkan cashback tinggi dan bonus poin.",
                "Kirimkan penawaran personal sesuai riwayat belanja.",
                "Dorong loyalitas lewat program tiering."
            ],
            "color": "green"
        },
        "regular": {
            "label": "Regular Customers",
            "description": "Pelanggan ini rutin belanja, tapi frekuensinya sedang dan nilai belanjanya belum tinggi.",
            "strategies": [
                "Gunakan promo bundling & hemat.",
                "Terapkan free-shipping threshold.",
                "Rekomendasikan produk untuk upselling & cross-selling.",
                "Berikan reward kecil untuk aktivitas rutin."
            ],
            "color": "blue"
        },
        "risk": {
            "label": "At-Risk / Inactive",
            "description": "Pelanggan ini lama tidak belanja, padahal sebelumnya cukup sering dan nilai belanjanya besar.",
            "strategies": [
                "Tawarkan voucher comeback & diskon 20-30%.",
                "Gunakan flash sale atau promo countdown.",
                "Kirimkan pengingat produk favorit mereka.",
                "Minta feedback singkat untuk perbaikan layanan."
            ],
            "color": "orange"
        }
    }

    # LOGIKA DINAMIS: Assign template ke Cluster ID berdasarkan statistik
    # 1. Cari Cluster 'At-Risk' (Paling lama tidak belanja -> Recency Tertinggi)
    at_risk_cluster = cluster_summary.loc[cluster_summary['Recency'].idxmax()]
    risk_id = int(at_risk_cluster['Cluster'])
    
    # 2. Sisa kluster (Active customers)
    active_clusters = cluster_summary[cluster_summary['Cluster'] != risk_id].copy()
    
    # 3. Urutkan active clusters berdasarkan Monetary (Kekayaan)
    #    Rank 1 (Terbesar) -> Best (Champions)
    #    Rank 2 (Menengah) -> Potential
    #    Rank 3 (Terbawah) -> Regular
    active_clusters = active_clusters.sort_values(by='Monetary', ascending=False)
    
    # Ambil ID berdasarkan urutan
    if len(active_clusters) >= 3:
        best_id = int(active_clusters.iloc[0]['Cluster'])
        potential_id = int(active_clusters.iloc[1]['Cluster'])
        regular_id = int(active_clusters.iloc[2]['Cluster'])
    else:
        # Fallback simple jika jumlah kluster < 4 (misal user ubah slider n_clusters)
        best_id = int(active_clusters.iloc[0]['Cluster']) if len(active_clusters) > 0 else -1
        potential_id = int(active_clusters.iloc[1]['Cluster']) if len(active_clusters) > 1 else -1
        regular_id = -1
    
    # Mapping Final: ID -> Template Key
    cluster_mapping = {}
    cluster_mapping[risk_id] = "risk"
    cluster_mapping[best_id] = "best"
    cluster_mapping[potential_id] = "potential"
    cluster_mapping[regular_id] = "regular"

    for index, row in cluster_summary.iterrows():
        cluster_id = int(row['Cluster'])
        count = rfm[rfm['Cluster'] == cluster_id].shape[0]
        
        # Ambil template berdasarkan mapping dinamis
        template_key = cluster_mapping.get(cluster_id, "regular") # Default ke regular jika bingung
        cluster_info = STRATEGY_TEMPLATES.get(template_key)
        
        # Fallback jika somehow None (should not happen with default logic)
        if not cluster_info:
             cluster_info = STRATEGY_TEMPLATES["regular"]
        
        with st.container():
            st.markdown(f"#### Kluster {cluster_id}: :{cluster_info['color']}[{cluster_info['label']}]")
            
            # Tampilkan deskripsi jika ada
            if cluster_info['description']:
                st.write(f"_{cluster_info['description']}_")
                
            col1, col2, col3, col4 = st.columns(4)
            col1.markdown(f"👥 **Total:** {count} User")
            col2.markdown(f"🕒 **Recency:** {row['Recency']:.0f} hari")
            col3.markdown(f"🛒 **Freq:** {row['Frequency']:.1f}x")
            col4.markdown(f"💰 **Uang:** Rp {row['Monetary']:,.0f}")
            
            # Format strategi menjadi list bullet points
            formatted_strategies = "\n".join([f"- {s}" for s in cluster_info['strategies']])
            st.info(f"**Strategi:**\n\n{formatted_strategies}")
            st.markdown("---")

    # Download Button
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="📥 Download Hasil (CSV)",
        data=rfm.to_csv(index=False).encode('utf-8'),
        file_name='hasil_segmentasi.csv',
        mime='text/csv',
    )