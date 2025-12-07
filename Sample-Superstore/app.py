import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Aplikasi Clustering",
    page_icon="📊",
    layout="wide"
)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("📌 Menu")
menu = st.sidebar.radio(
    "Pilih Menu",
    ["Upload Data", "Preprocessing", "Clustering", "Visualisasi"]
)

st.sidebar.markdown("---")
st.sidebar.info("Aplikasi Clustering dengan Streamlit")

# ===============================
# UPLOAD DATA
# ===============================
if menu == "Upload Data":
    st.title("📂 Upload Dataset")

    file = st.file_uploader(
        "Upload file CSV",
        type=["csv"]
    )

    if file is not None:
        df = pd.read_csv(file)
        st.session_state["data"] = df

        st.success("Dataset berhasil diupload ✅")
        st.dataframe(df.head())

# ===============================
# PREPROCESSING
# ===============================
elif menu == "Preprocessing":
    st.title("⚙️ Preprocessing Data")

    if "data" not in st.session_state:
        st.warning("Silakan upload dataset terlebih dahulu")
    else:
        df = st.session_state["data"]

        st.write("Data Awal")
        st.dataframe(df.head())

        # Pilih kolom numerik
        numeric_cols = df.select_dtypes(include=np.number).columns
        selected_cols = st.multiselect(
            "Pilih kolom numerik",
            numeric_cols,
            default=numeric_cols
        )

        if st.button("Standarisasi Data"):
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[selected_cols])

            st.session_state["scaled_data"] = scaled_data
            st.session_state["selected_cols"] = selected_cols

            st.success("Data berhasil distandarisasi ✅")

# ===============================
# CLUSTERING
# ===============================
elif menu == "Clustering":
    st.title("🧠 Proses Clustering")

    if "scaled_data" not in st.session_state:
        st.warning("Lakukan preprocessing terlebih dahulu")
    else:
        k = st.slider(
            "Jumlah Cluster (K)",
            min_value=2,
            max_value=10,
            value=3
        )

        if st.button("Jalankan K-Means"):
            model = KMeans(
                n_clusters=k,
                random_state=42
            )

            labels = model.fit_predict(
                st.session_state["scaled_data"]
            )

            df = st.session_state["data"].copy()
            df["Cluster"] = labels

            st.session_state["clustered_df"] = df

            st.success("Clustering selesai ✅")
            st.dataframe(df.head())

# ===============================
# VISUALISASI
# ===============================
elif menu == "Visualisasi":
    st.title("📊 Visualisasi Hasil Clustering")

    if "clustered_df" not in st.session_state:
        st.warning("Lakukan clustering terlebih dahulu")
    else:
        df = st.session_state["clustered_df"]
        cols = st.session_state["selected_cols"]

        if len(cols) < 2:
            st.warning("Pilih minimal 2 kolom numerik")
        else:
            x_col = st.selectbox("Sumbu X", cols, index=0)
            y_col = st.selectbox("Sumbu Y", cols, index=1)

            fig, ax = plt.subplots()
            scatter = ax.scatter(
                df[x_col],
                df[y_col],
                c=df["Cluster"],
                cmap="viridis"
            )

            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title("Visualisasi Cluster")

            st.pyplot(fig)
