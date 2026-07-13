import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
    <style>
        /* Global styling */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        /* Main title styling */
        .main-title {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            color: white !important;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .main-title h1 {
            color: white !important;
            font-weight: 700;
        }
        
        .main-title p {
            color: rgba(255,255,255,0.9) !important;
            font-size: 1.1rem;
        }
        
        /* Card styling */
        .card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.3);
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        
        /* Metric card styling */
        .metric-card {
            background: white;
            padding: 1.2rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            border-left: 4px solid #667eea;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Form styling */
        .stForm {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 25px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Dataframe styling */
        .dataframe-container {
            background: white;
            border-radius: 15px;
            padding: 0.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            overflow: hidden;
        }
        
        /* Subheader styling */
        .custom-subheader {
            color: #2c3e50;
            font-weight: 600;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #667eea;
            margin-bottom: 1.5rem;
        }
        
        /* Prediction result styling */
        .prediction-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-top: 1rem;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .prediction-box h3 {
            color: white !important;
            margin: 0;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            padding: 1.5rem;
            background: white;
            border-radius: 15px;
            margin-top: 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
    </style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown("""
    <div class="main-title">
        <h1>📈 Regresi Linear – Dataset Sample Superstore</h1>
        <p>Prediksi Sales menggunakan Quantity, Discount, dan Profit</p>
    </div>
""", unsafe_allow_html=True)

# ===============================
# LOAD DATASET (PATH AMAN STREAMLIT CLOUD)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Sample - Superstore.csv")

if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset tidak ditemukan. Pastikan file 'Sample - Superstore.csv' berada satu folder dengan app.py")
    st.stop()

df = pd.read_csv(DATA_PATH, encoding="latin1")
df.columns = df.columns.str.strip()

# ===============================
# PREVIEW DATA
# ===============================
st.markdown('<div class="custom-subheader">📊 Preview Dataset</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# SELEKSI DATA REGRESI
# ===============================
st.markdown('<div class="custom-subheader">📌 Data yang Digunakan</div>', unsafe_allow_html=True)

df_reg = df[['Quantity', 'Discount', 'Profit', 'Sales']].dropna()
st.info(f"✅ Jumlah data yang digunakan: **{df_reg.shape[0]}** baris")

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
st.markdown('<div class="custom-subheader">📉 Evaluasi Model Regresi</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 MAE</div>
            <div class="metric-value">{mean_absolute_error(y_test, y_pred):,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card" style="border-left-color: #e74c3c;">
            <div class="metric-label">📈 MSE</div>
            <div class="metric-value">{mean_squared_error(y_test, y_pred):,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card" style="border-left-color: #f39c12;">
            <div class="metric-label">🎯 RMSE</div>
            <div class="metric-value">{np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="metric-card" style="border-left-color: #27ae60;">
            <div class="metric-label">⭐ R² Score</div>
            <div class="metric-value">{r2_score(y_test, y_pred):.3f}</div>
        </div>
    """, unsafe_allow_html=True)

# ===============================
# KOEFISIEN REGRESI
# ===============================
st.markdown('<div class="custom-subheader">📐 Koefisien Regresi Linear</div>', unsafe_allow_html=True)

coef_df = pd.DataFrame({
    "Variabel": X.columns,
    "Koefisien": model.coef_
})

# Styling koefisien dataframe
def color_coef(val):
    color = '#27ae60' if val > 0 else '#e74c3c'
    return f'color: {color}; font-weight: 600'

st.dataframe(
    coef_df.style.applymap(color_coef, subset=['Koefisien']).format({'Koefisien': '{:.4f}'}),
    use_container_width=True,
    hide_index=True
)

# ===============================
# VISUALISASI
# ===============================
st.markdown('<div class="custom-subheader">📊 Perbandingan Sales Aktual vs Prediksi</div>', unsafe_allow_html=True)

# Set style seaborn
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot with regression line
scatter = ax.scatter(y_test, y_pred, alpha=0.6, c=y_test, cmap='viridis', edgecolors='black', linewidth=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')

ax.set_xlabel("Sales Aktual", fontsize=12, fontweight='bold')
ax.set_ylabel("Sales Prediksi", fontsize=12, fontweight='bold')
ax.set_title("Sales Aktual vs Sales Prediksi (Regresi Linear)", fontsize=14, fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Sales Aktual', fontsize=10)

ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

st.pyplot(fig, use_container_width=True)

# ===============================
# FORM PREDIKSI BARU
# ===============================
st.markdown('<div class="custom-subheader">🔮 Prediksi Sales Baru</div>', unsafe_allow_html=True)

with st.form("prediksi_sales", clear_on_submit=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        qty = st.number_input(
            "📦 Quantity", 
            min_value=1, 
            value=5,
            help="Jumlah produk yang dibeli"
        )
    
    with col2:
        disc = st.slider(
            "🏷️ Discount", 
            0.0, 1.0, 0.2, 0.05,
            help="Diskon dalam persentase (0.0 - 1.0)"
        )
    
    with col3:
        profit = st.number_input(
            "💰 Profit", 
            value=50.0,
            help="Keuntungan dari penjualan"
        )
    
    submit = st.form_submit_button("🚀 Prediksi Sales", use_container_width=True)

if submit:
    hasil = model.predict([[qty, disc, profit]])
    st.markdown(f"""
        <div class="prediction-box">
            <h3>💰 Prediksi Sales: ${hasil[0]:,.2f}</h3>
            <p style="margin-top: 0.5rem; opacity: 0.9;">
                Berdasarkan Quantity: {qty}, Discount: {disc:.0%}, Profit: ${profit:,.2f}
            </p>
        </div>
    """, unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================
st.markdown("""
    <div class="footer">
        <p style="margin: 0; color: #7f8c8d;">
            📊 Model: Regresi Linear | Dataset: Sample Superstore | Dibangun dengan ❤️ menggunakan Streamlit
        </p>
        <p style="margin: 0.5rem 0 0 0; color: #95a5a6; font-size: 0.85rem;">
            © 2026 - Semua Hak Dilindungi
        </p>
    </div>
""", unsafe_allow_html=True)
