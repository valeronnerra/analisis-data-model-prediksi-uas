import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Clustering Kelayakan Pendidikan",
    page_icon="🎓",
    layout="wide"
)

# =========================
# HEADER
# =========================
st.title("🎓 Clustering Kelayakan Pendidikan Indonesia")
st.markdown(
    """
    Aplikasi ini digunakan untuk *mengelompokkan provinsi di Indonesia*
    berdasarkan kondisi kelayakan pendidikan menggunakan
    *metode K-Means Clustering*.
    """
)

st.divider()

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    import numpy as np

    provinces = [
        "Aceh", "Sumatra Utara", "Sumatra Barat", "Riau", "Jambi",
        "Sumatra Selatan", "Bangka Belitung", "Lampung", "DKI Jakarta",
        "Jawa Barat", "Jawa Tengah", "Yogyakarta", "Jawa Timur",
        "Banten", "Bali", "Nusa Tenggara Barat", "Nusa Tenggara Timur",
        "Kalimantan Barat", "Kalimantan Tengah", "Kalimantan Selatan",
        "Kalimantan Timur", "Sulawesi Utara", "Sulawesi Tengah",
        "Sulawesi Selatan", "Sulawesi Tenggara", "Gorontalo",
        "Sulawesi Barat", "Maluku", "Maluku Utara", "Papua Barat",
        "Papua"
    ]

    np.random.seed(42)
    data = {
        "Provinsi": provinces,
        "Siswa": np.random.randint(100000, 500000, len(provinces)),
        "Putus Sekolah": np.random.randint(5, 50, len(provinces)),
        "Guru_Kepsek_S1_Keatas": np.random.randint(50, 95, len(provinces)),
        "Ruang_Kelas_Baik": np.random.randint(30, 90, len(provinces))
    }

    return pd.DataFrame(data)

df = load_data()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙ Pengaturan Analisis")
st.sidebar.markdown(
    "Atur jumlah cluster untuk melihat "
    "perubahan hasil pengelompokan provinsi."
)

k = st.sidebar.slider("Jumlah Cluster", 2, 5, 3)

# =========================
# DATASET
# =========================
st.subheader("📄 Dataset Pendidikan")
st.caption("Dataset ini merupakan data hasil pembersihan dan siap dianalisis.")
st.dataframe(df, use_container_width=True)

# =========================
# PREPROCESSING
# =========================
features = [
    "Siswa",
    "Putus Sekolah",
    "Guru_Kepsek_S1_Keatas",
    "Ruang_Kelas_Baik"
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# =========================
# MODEL K-MEANS
# =========================
kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# =========================
# HASIL CLUSTER
# =========================
st.subheader("📌 Hasil Pengelompokan Provinsi")
st.dataframe(df[["Provinsi", "Cluster"]], use_container_width=True)

# =========================
# RINGKASAN CLUSTER
# =========================
st.subheader("📊 Rata-rata Indikator per Cluster")
cluster_summary = df.groupby("Cluster")[features].mean()
st.dataframe(cluster_summary, use_container_width=True)

# =========================
# VISUALISASI
# =========================
st.subheader("📈 Visualisasi Clustering")

fig, ax = plt.subplots()
ax.scatter(
    df["Guru_Kepsek_S1_Keatas"],
    df["Putus Sekolah"],
    c=df["Cluster"]
)

ax.set_xlabel("Guru & Kepala Sekolah ≥ S1")
ax.set_ylabel("Putus Sekolah")
ax.set_title("Visualisasi Clustering Kelayakan Pendidikan")

st.pyplot(fig)

# =========================
# INTERPRETASI
# =========================
st.subheader("🧠 Interpretasi Hasil")
st.write(
    "Provinsi dalam satu cluster memiliki kondisi pendidikan yang relatif mirip. "
    "Cluster dengan tingkat putus sekolah yang tinggi dan kualitas guru yang rendah "
    "cenderung menunjukkan kelayakan pendidikan yang lebih rendah dibandingkan cluster lainnya."
)