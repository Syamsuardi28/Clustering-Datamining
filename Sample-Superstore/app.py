import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Regresi Linear Superstore",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Regresi Linear – Dataset Sample Superstore")
st.caption("Prediksi Sales menggunakan Quantity, Discount, dan Profit")

# ===============================
# LOAD DATASET (PATH AMAN STREAMLIT CLOUD)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Sample - Superstore.csv")

if not os.path.exists(DATA_PATH):
    st.error("Dataset tidak ditemukan. Pastikan file 'Sample - Superstore.csv' berada satu folder dengan app.py")
    st.stop()

df = pd.read_csv(DATA_PATH, encoding="latin1")
df.columns = df.columns.str.strip()

# ===============================
# PREVIEW DATA
# ===============================
st.subheader("📊 Preview Dataset")
st.dataframe(df.head())

# ===============================
# SELEKSI DATA REGRESI
# ===============================
st.subheader("📌 Data yang Digunakan")

df_reg = df[['Quantity', 'Discount', 'Profit', 'Sales']].dropna()
st.write(f"Jumlah data digunakan: {df_reg.shape[0]} baris")

X = df_reg[['Quantity', 'Discount', 'Profit']]
y = df_reg['Sales']

# ===============================
# SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# TRAIN MODEL
# ===============================
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ===============================
# EVALUASI MODEL
# ===============================
st.subheader("📉 Evaluasi Model Regresi")

col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):,.2f}")
col2.metric("MSE", f"{mean_squared_error(y_test, y_pred):,.2f}")
col3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
col4.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")

# ===============================
# KOEFISIEN REGRESI
# ===============================
st.subheader("📐 Koefisien Regresi Linear")

coef_df = pd.DataFrame({
    "Variabel": X.columns,
    "Koefisien": model.coef_
})

st.table(coef_df)

# ===============================
# VISUALISASI
# ===============================
st.subheader("📊 Perbandingan Sales Aktual vs Prediksi")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel("Sales Aktual")
ax.set_ylabel("Sales Prediksi")
ax.set_title("Sales Aktual vs Sales Prediksi (Regresi Linear)")
st.pyplot(fig)

# ===============================
# FORM PREDIKSI BARU
# ===============================
st.subheader("🔮 Prediksi Sales Baru")

with st.form("prediksi_sales"):
    qty = st.number_input("Quantity", min_value=1, value=5)
    disc = st.slider("Discount", 0.0, 1.0, 0.2, 0.05)
    profit = st.number_input("Profit", value=50.0)
    submit = st.form_submit_button("Prediksi")

if submit:
    hasil = model.predict([[qty, disc, profit]])
    st.success(f"Prediksi Sales: ${hasil[0]:,.2f}")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("Model: Regresi Linear | Dataset: Sample Superstore | Streamlit App")
