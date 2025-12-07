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
# SIDEBAR (DIPERBAGUS)
# ===============================
with st.sidebar:
    st.markdown("## 📊 Clustering App")
    st.caption("Data Mining dengan Streamlit")
    st.markdown("---")

    menu = st.radio(
        "📌 Menu Utama",
        ["Upload Data", "Preprocessing", "Clustering", "Visualisasi"]
    )

    st.markdown("---")

    if "data" in st.session_state:
        st.success("✅ Dataset dimuat")
        st.caption(f"Jumlah data: {st.session_state['data'].shape[0]} baris")
    else:
        st.warning("⚠️ Dataset belum diupload")

    st.markdown("---")
    st.caption("👨‍💻 Dibuat dengan Streamlit")

# ===============================
# UPLOAD DATA
# ===============================
if menu == "Upload Data":
    st.title("📂 Upload Dataset CSV")

    file = st.file_uploader(
        "Upload file CSV",
        type=["csv"]
    )

    if file is not None:
        # ✅ FIX: UnicodeDecodeError
        try:
            df = pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding="latin1")

        st.session_state["data"] = df

        st.success("✅ Dataset berhasil diupload")
        st.dataframe(df.head(), use_container_width=True)

# ===============================
# PREPROCESSING
# ===============================
elif menu == "Preprocessing":
    st.title("⚙️ Preprocessing Data")

    if "data" not in st.session_state:
        st.warning("Silakan upload dataset terlebih dahulu")
    else:
        df = st.session_state["data"]

        st.subheader("📄 Data Awal")
        st.dataframe(df.head(), use_container_width=True)

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            st.error("❌ Dataset tidak memiliki kolom numerik")
        else:
            selected_cols = st.multiselect(
                "Pilih kolom numerik",
                numeric_cols,
                default=numeric_cols
            )

            if st.button("🔄 Standarisasi Data"):
                if len(selected_cols) == 0:
                    st.warning("Pilih minimal satu kolom")
                else:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(df[selected_cols])

                    st.session_state["scaled_data"] = scaled_data
                    st.session_state["selected_cols"] = selected_cols

                    st.success("✅ Data berhasil distandarisasi")

# ===============================
# CLUSTERING
# ===============================
elif menu == "Clustering":
    st.title("🧠 Proses Clustering (K-Means)")

    if "scaled_data" not in st.session_state:
        st.warning("Lakukan preprocessing terlebih dahulu")
    else:
        k = st.slider(
            "Jumlah Cluster (K)",
            min_value=2,
            max_value=10,
            value=3
        )

        if st.button("🚀 Jalankan K-Means"):
            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10  # ✅ FIX WARNING
            )

            labels = model.fit_predict(st.session_state["scaled_data"])

            df = st.session_state["data"].copy()
            df["Cluster"] = labels

            st.session_state["clustered_df"] = df

            st.success("✅ Clustering selesai")
            st.dataframe(df.head(), use_container_width=True)

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
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Sumbu X", cols, index=0)
            with col2:
                y_col = st.selectbox("Sumbu Y", cols, index=1)

            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                df[x_col],
                df[y_col],
                c=df["Cluster"],
                cmap="viridis",
                alpha=0.8
            )

            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title("Visualisasi Cluster")

            st.pyplot(fig)
