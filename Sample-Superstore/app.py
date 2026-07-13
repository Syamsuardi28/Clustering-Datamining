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
            padding: 1.5rem 0;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            color: white !important;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .main-title h1 {
            color: white !important;
            font-weight: 700;
            margin: 0;
        }
        
        .main-title p {
            color: rgba(255,255,255,0.9) !important;
            font-size: 1.1rem;
            margin: 0.5rem 0 0 0;
        }
        
        /* Card styling */
        .card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        
        /* Metric card styling */
        .metric-card {
            background: white;
            padding: 1.2rem 1rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            border-left: 4px solid #667eea;
            transition: all 0.3s ease;
            height: 100%;
        }
        
        .metric-card:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin: 0.3rem 0;
        }
        
        .metric-label {
            font-size: 0.85rem;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }
        
        .metric-icon {
            font-size: 1.5rem;
        }
        
        /* Dataframe styling */
        .dataframe-container {
            background: white;
            border-radius: 15px;
            padding: 0.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            overflow: hidden;
            border: 1px solid #e8ecf1;
        }
        
        /* Subheader styling */
        .custom-subheader {
            color: #2c3e50;
            font-weight: 600;
            font-size: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #667eea;
            margin: 1.5rem 0 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
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
            animation: slideIn 0.5s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .prediction-box h3 {
            color: white !important;
            margin: 0;
            font-size: 1.8rem;
        }
        
        .prediction-box p {
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }
        
        /* Form styling */
        .stForm {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border: 1px solid #e8ecf1;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            border: none;
            padding: 0.6rem 2rem;
            border-radius: 25px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            color: white;
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }
        
        /* Input styling */
        .stNumberInput > div > div > input, 
        .stSlider > div > div > div {
            border-radius: 8px;
        }
        
        /* Info box styling */
        .stAlert {
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            padding: 1.5rem;
            background: white;
            border-radius: 15px;
            margin-top: 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            border: 1px solid #e8ecf1;
        }
        
        /* Container padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Table styling */
        .dataframe {
            font-size: 0.9rem;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .metric-value {
                font-size: 1.3rem;
            }
            .main-title h1 {
                font-size: 1.5rem;
            }
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

col_info1, col_info2 = st.columns([1, 3])
with col_info1:
    st.info(f"**{df_reg.shape[0]}** baris")
with col_info2:
    st.info("✅ Data siap untuk proses regresi linear")

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

# Hitung metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="metric-card" style="border-left-color: #3498db;">
            <div class="metric-icon">📊</div>
            <div class="metric-label">MAE</div>
            <div class="metric-value">{mae:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card" style="border-left-color: #e74c3c;">
            <div class="metric-icon">📈</div>
            <div class="metric-label">MSE</div>
            <div class="metric-value">{mse:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card" style="border-left-color: #f39c12;">
            <div class="metric-icon">🎯</div>
            <div class="metric-label">RMSE</div>
            <div class="metric-value">{rmse:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="metric-card" style="border-left-color: #27ae60;">
            <div class="metric-icon">⭐</div>
            <div class="metric-label">R² Score</div>
            <div class="metric-value">{r2:.3f}</div>
        </div>
    """, unsafe_allow_html=True)

# ===============================
# KOEFISIEN REGRESI
# ===============================
st.markdown('<div class="custom-subheader">📐 Koefisien Regresi Linear</div>', unsafe_allow_html=True)

coef_df = pd.DataFrame({
    "Variabel": X.columns,
    "Koefisien": model.coef_,
    "Interpretasi": ["Positif ↑" if c > 0 else "Negatif ↓" for c in model.coef_]
})

# Styling koefisien dengan method yang kompatibel
def highlight_coef(val):
    if isinstance(val, (int, float)):
        if val > 0:
            return 'background-color: #d4edda; color: #155724; font-weight: 600'
        elif val < 0:
            return 'background-color: #f8d7da; color: #721c24; font-weight: 600'
    return ''

def highlight_interpretasi(val):
    if "Positif" in val:
        return 'background-color: #d4edda; color: #155724; font-weight: 600'
    else:
        return 'background-color: #f8d7da; color: #721c24; font-weight: 600'

# Gunakan styler dengan method yang benar
styled_df = coef_df.style.format({'Koefisien': '{:.4f}'})

# Apply styling per kolom
styled_df = styled_df.apply(lambda x: [highlight_coef(v) for v in x], subset=['Koefisien'])
styled_df = styled_df.apply(lambda x: [highlight_interpretasi(v) for v in x], subset=['Interpretasi'])

st.dataframe(styled_df, use_container_width=True, hide_index=True)

# ===============================
# VISUALISASI
# ===============================
st.markdown('<div class="custom-subheader">📊 Perbandingan Sales Aktual vs Prediksi</div>', unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

# Scatter plot
scatter = ax.scatter(y_test, y_pred, 
                    alpha=0.6, 
                    c=y_test, 
                    cmap='RdYlGn', 
                    edgecolors='black', 
                    linewidth=0.5,
                    s=50)

# Perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 
        'r--', lw=2, label='Perfect Prediction', alpha=0.7)

# Set labels dan title
ax.set_xlabel("Sales Aktual", fontsize=12, fontweight='bold')
ax.set_ylabel("Sales Prediksi", fontsize=12, fontweight='bold')
ax.set_title("Sales Aktual vs Sales Prediksi (Regresi Linear)", 
             fontsize=14, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle='--')

# Legend
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Sales Aktual', fontsize=10, fontweight='bold')

# Tambahkan R² pada plot
ax.text(0.02, 0.98, f'R² = {r2:.3f}', 
        transform=ax.transAxes, 
        fontsize=12, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
st.pyplot(fig, use_container_width=True)

# ===============================
# FORM PREDIKSI BARU
# ===============================
st.markdown('<div class="custom-subheader">🔮 Prediksi Sales Baru</div>', unsafe_allow_html=True)

with st.form("prediksi_sales", clear_on_submit=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📦 Quantity")
        qty = st.number_input(
            "Jumlah produk", 
            min_value=1, 
            max_value=100,
            value=5,
            step=1,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### 🏷️ Discount")
        disc = st.slider(
            "Diskon", 
            0.0, 1.0, 0.2, 0.05,
            format="%.0f%%",
            label_visibility="collapsed"
        )
    
    with col3:
        st.markdown("### 💰 Profit")
        profit = st.number_input(
            "Keuntungan", 
            min_value=-1000.0,
            max_value=10000.0,
            value=50.0,
            step=10.0,
            label_visibility="collapsed"
        )
    
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        submit = st.form_submit_button("🚀 Prediksi Sales", use_container_width=True)

if submit:
    hasil = model.predict([[qty, disc, profit]])
    st.markdown(f"""
        <div class="prediction-box">
            <h3>💰 ${hasil[0]:,.2f}</h3>
            <p>Prediksi Sales berdasarkan:</p>
            <p style="font-size: 0.9rem;">
                Quantity: {qty} | Discount: {disc:.0%} | Profit: ${profit:,.2f}
            </p>
        </div>
    """, unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================
st.markdown("""
    <div class="footer">
        <p style="margin: 0; color: #2c3e50; font-weight: 500;">
            📊 Model: Regresi Linear | Dataset: Sample Superstore
        </p>
        <p style="margin: 0.5rem 0 0 0; color: #95a5a6; font-size: 0.85rem;">
            Dibangun dengan ❤️ menggunakan Streamlit
        </p>
        <p style="margin: 0.2rem 0 0 0; color: #bdc3c7; font-size: 0.75rem;">
            © 2026 - Semua Hak Dilindungi
        </p>
    </div>
""", unsafe_allow_html=True)
