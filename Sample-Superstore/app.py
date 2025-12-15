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
    page_title="Regresi Sales Superstore",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Regresi Linear – Dataset Sample Superstore")
st.caption("Prediksi Sales menggunakan Quantity, Discount, dan Profit")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv(
        "Sample - Superstore.csv",
        encoding="latin1"
    )

df = load_data()
df.columns = df.columns.str.strip()

# ===============================
# PREVIEW DATA
# ===============================
st.subheader("📊 Preview Dataset")
st.dataframe(df.head())

# ===============================
# SELEKSI DATA REGRESI
# ===============================
df_reg = df[['Quantity', 'Discount', 'Profit', 'Sales']].dropna()

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

# ===============================
# PREDIKSI
# ===============================
y_pred = model.predict(X_test)

# ===============================
# EVALUASI MODEL
# ===============================
st.subheader("📉 Evaluasi Model")

col1, col2, col3, col4 = st.columns(4)

col1.metric("MAE", round(mean_absolute_error(y_test, y_pred), 2))
col2.metric("MSE", round(mean_squared_error(y_test, y_pred), 2))
col3.metric("RMSE", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
col4.metric("R² Score", round(r2_score(y_test, y_pred), 3))

# ===============================
# KOEFISIEN REGRESI
# ===============================
st.subheader("📐 Koefisien Regresi")

coef_df = pd.DataFrame({
    "Variabel": X.columns,
    "Koefisien": model.coef_
})

st.table(coef_df)

# ===============================
# VISUALISASI
# ===============================
st.subheader("📊 Sales Aktual vs Prediksi")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel("Sales Aktual")
ax.set_ylabel("Sales Prediksi")
ax.set_title("Perbandingan Sales Aktual dan Prediksi")

st.pyplot(fig)

# ===============================
# FORM PREDIKSI
# ===============================
st.subheader("🔮 Prediksi Sales Baru")

with st.form("prediksi_form"):
    qty = st.number_input("Quantity", min_value=1, value=5)
    disc = st.slider("Discount", 0.0, 1.0, 0.2, 0.05)
    profit = st.number_input("Profit", value=50.0)

    submit = st.form_submit_button("Prediksi")

if submit:
    hasil = model.predict([[qty, disc, profit]])
    st.success(f"Prediksi Sales: ${hasil[0]:,.2f}")
