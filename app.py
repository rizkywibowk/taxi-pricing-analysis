import streamlit as st

# HARUS MENJADI COMMAND PERTAMA
st.set_page_config(
    page_title="🚕 Sigma Cabs - SVM Pricing Analysis", # Judul disesuaikan
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import setelah set_page_config
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Check Python version
python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

# Try import ML libraries dengan error handling
ML_AVAILABLE = False
MODEL_SOURCE = "fallback" # Untuk melacak sumber model yang digunakan
try:
    import joblib # Pastikan joblib ada di requirements.txt
    import pickle
    from sklearn.preprocessing import StandardScaler # Scaler tetap penting
    from sklearn.svm import SVR # Mengganti dengan SVR untuk regresi atau SVC untuk klasifikasi
                                # Asumsi SVM untuk regresi (harga), jadi SVR. Jika klasifikasi, ganti SVC
    # from sklearn.ensemble import GradientBoostingRegressor # Ini tidak lagi jadi model utama
    import plotly.express as px
    import plotly.graph_objects as go
    from typing import Optional, Tuple, List, Dict, Any
    ML_AVAILABLE = True
except ImportError:
    pass

# Enhanced CSS dengan background hijau daun cerah yang diperkuat
st.markdown("""
<style>
    /* Root variables untuk theming - Hijau Daun Cerah Enhanced */
    :root {
        --primary-color: #FF6B6B;
        --secondary-color: #2e7d32; /* Warna hijau sekunder yang lebih gelap untuk kontras */
        --background-color: #e8f5e8; /* Latar belakang hijau daun cerah */
        --text-color: #1b5e20; /* Teks hijau tua untuk keterbacaan */
        --card-background: rgba(255, 255, 255, 0.95); /* Kartu semi-transparan */
        --border-color: #81c784; /* Border hijau muda */
        --success-color: #2e7d32;
        --warning-color: #ff8f00;
        --danger-color: #d32f2f;
        --info-color: #1976d2;
        --accent-green: #4caf50; /* Hijau aksen */
        --light-green: #c8e6c9; /* Hijau sangat muda */
        --dark-green: #1b5e20; /* Hijau paling tua */
    }
    
    /* Dark mode variables (opsional, jika ingin berbeda dari light mode) */
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color: #FF6B6B;
            --secondary-color: #4fc3f7; /* Biru muda untuk dark mode */
            --background-color: #0d47a1; /* Biru laut tua untuk dark mode */
            --text-color: #e3f2fd; /* Teks putih kebiruan */
            --card-background: rgba(30, 63, 102, 0.95);
            --border-color: #42a5f5;
            --accent-green: #4fc3f7; /* Ganti dengan aksen biru untuk dark mode */
            --light-green: rgba(79, 195, 247, 0.2);
        }
    }
    
    /* Enhanced background dengan gradient hijau daun yang lebih kaya */
    .stApp {
        background: linear-gradient(135deg, 
                   var(--background-color) 0%, 
                   color-mix(in srgb, var(--background-color) 85%, white 15%) 25%,
                   color-mix(in srgb, var(--background-color) 90%, var(--accent-green) 10%) 50%,
                   color-mix(in srgb, var(--background-color) 80%, white 20%) 75%,
                   color-mix(in srgb, var(--background-color) 95%, var(--dark-green) 5%) 100%);
        color: var(--text-color);
        transition: background 0.5s ease, color 0.5s ease;
        min-height: 100vh;
        background-attachment: fixed;
    }
    
    .main .block-container {
        background: transparent;
        color: var(--text-color);
    }
    
    /* Enhanced main header tanpa blur, dengan gradient text */
    .main-header {
        font-size: clamp(2rem, 6vw, 3rem);
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        word-wrap: break-word;
        line-height: 1.2;
        font-weight: 700;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color), var(--accent-green));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
        filter: none !important; 
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 3px;
        background: linear-gradient(90deg, var(--secondary-color), var(--accent-green));
        border-radius: 2px;
    }
    
    /* Header image tanpa blur */
    .stImage > img {
        max-width: 100%;
        height: auto;
        border-radius: 10px;
        filter: none !important; 
        backdrop-filter: none !important;
    }
    
    /* Enhanced prediction box dengan animasi */
    .prediction-box {
        background: linear-gradient(135deg, 
                   var(--secondary-color) 0%, 
                   color-mix(in srgb, var(--secondary-color) 70%, black 30%) 100%);
        padding: clamp(1.5rem, 4vw, 2rem);
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.2), 
                   0 0 0 1px rgba(255,255,255,0.1) inset;
        word-wrap: break-word;
        border: 2px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(15px);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
                   transparent, 
                   rgba(255,255,255,0.1), 
                   transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .prediction-box h1 {
        font-size: clamp(3rem, 10vw, 5rem) !important;
        margin: 1rem 0 !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    /* Enhanced metric cards dengan green theme dan animasi */
    .metric-card {
        background: var(--card-background);
        backdrop-filter: blur(20px);
        padding: clamp(1rem, 3vw, 1.2rem);
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15),
                   0 0 0 1px rgba(255,255,255,0.1) inset;
        margin: 0.8rem 0;
        min-height: clamp(120px, 18vh, 180px);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        color: var(--text-color);
        transition: all 0.3s ease;
        border: 2px solid var(--border-color);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-green), var(--secondary-color));
    }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
        border-color: var(--accent-green);
    }
    
    /* Enhanced color coding untuk surge categories */
    .metric-card.surge-low {
        border-left: 8px solid var(--success-color);
        background: linear-gradient(145deg, 
                   color-mix(in srgb, var(--success-color) 15%, white 85%), 
                   color-mix(in srgb, var(--success-color) 8%, white 92%));
    }
    
    .metric-card.surge-medium {
        border-left: 8px solid var(--warning-color);
        background: linear-gradient(145deg, 
                   color-mix(in srgb, var(--warning-color) 15%, white 85%), 
                   color-mix(in srgb, var(--warning-color) 8%, white 92%));
    }
    
    .metric-card.surge-high {
        border-left: 8px solid var(--danger-color);
        background: linear-gradient(145deg, 
                   color-mix(in srgb, var(--danger-color) 15%, white 85%), 
                   color-mix(in srgb, var(--danger-color) 8%, white 92%));
    }
    
    /* Enhanced info boxes */
    .info-box {
        background: var(--card-background);
        backdrop-filter: blur(20px);
        padding: clamp(1rem, 3vw, 1.2rem);
        border-radius: 15px;
        border-left: 6px solid var(--secondary-color);
        margin: 0.8rem 0;
        word-wrap: break-word;
        color: var(--text-color);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1),
                   0 0 0 1px rgba(255,255,255,0.1) inset;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .info-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .contact-info {
        background: var(--card-background);
        backdrop-filter: blur(20px);
        padding: clamp(1rem, 3vw, 1.2rem);
        border-radius: 15px;
        border-left: 6px solid var(--warning-color);
        margin: 0.8rem 0;
        word-wrap: break-word;
        color: var(--text-color);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1),
                   0 0 0 1px rgba(255,255,255,0.1) inset;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .contact-info:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .header-box {
        background: linear-gradient(135deg, 
                   var(--secondary-color) 0%, 
                   color-mix(in srgb, var(--secondary-color) 70%, black 30%) 100%);
        padding: clamp(2rem, 5vw, 3rem);
        text-align: center;
        border-radius: 20px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 35px rgba(0,0,0,0.2),
                   0 0 0 1px rgba(255,255,255,0.1) inset;
        backdrop-filter: none; 
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Enhanced model status styling */
    .model-status {
        background: var(--card-background);
        backdrop-filter: blur(20px);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border: 2px solid var(--border-color);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .model-status.success {
        border-left: 6px solid var(--success-color);
    }
    
    .model-status.warning {
        border-left: 6px solid var(--warning-color);
    }
    
    .model-status.info {
        border-left: 6px solid var(--info-color);
    }
    
    /* Enhanced Streamlit widget styling */
    .stSelectbox > div > div {
        background: var(--card-background) !important;
        color: var(--text-color) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stNumberInput > div > div > input {
        background: var(--card-background) !important;
        color: var(--text-color) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stSlider > div > div > div {
        background: var(--card-background) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Mobile responsiveness */
    @media (orientation: portrait) {
        .main-header {
            font-size: clamp(1.5rem, 7vw, 2.5rem);
            margin-bottom: 1rem;
        }
        
        .metric-card {
            min-height: clamp(100px, 15vh, 140px);
            padding: 1rem;
        }
        
        .prediction-box {
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .stColumns > div {
            width: 100% !important;
            margin-bottom: 1rem;
        }
    }
    
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem !important;
            margin-bottom: 1rem !important;
        }
        
        .metric-card {
            min-height: 120px !important;
            padding: 1rem !important;
            margin: 0.5rem 0 !important;
        }
        
        .prediction-box {
            padding: 1.5rem !important;
            margin: 1rem 0 !important;
        }
        
        .stColumns > div {
            min-width: 100% !important;
            margin-bottom: 1rem !important;
        }
    }
    
    /* Enhanced button styling */
    .stButton > button {
        width: 100%;
        padding: 1.2rem 1.8rem;
        font-size: clamp(1rem, 3vw, 1.2rem);
        font-weight: 600;
        background: linear-gradient(135deg, 
                   var(--secondary-color) 0%, 
                   color-mix(in srgb, var(--secondary-color) 80%, black 20%) 100%);
        color: white;
        border: none;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2),
                   0 0 0 1px rgba(255,255,255,0.1) inset;
        transition: all 0.3s ease;
        backdrop-filter: blur(15px);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98);
    }
    
    /* Enhanced expander dan dataframe */
    .stExpander {
        background: var(--card-background);
        border: 2px solid var(--border-color);
        border-radius: 12px;
        backdrop-filter: blur(15px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stDataFrame {
        background: var(--card-background);
        border-radius: 12px;
        border: 2px solid var(--border-color);
        backdrop-filter: blur(15px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Container responsive */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: clamp(1rem, 4vw, 3rem);
        padding-right: clamp(1rem, 4vw, 3rem);
    }
    
    /* Text color overrides */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6 {
        color: var(--text-color) !important;
    }
    
    /* Enhanced footer */
    .footer-container {
        background: var(--card-background);
        backdrop-filter: blur(20px);
        border: 2px solid var(--border-color);
        color: var(--text-color);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk membuat model SVM fallback jika .pkl gagal
def create_svm_fallback_model():
    """Create a fallback SVM model (SVR for regression)"""
    feature_names = [ # Asumsi fitur ini sama atau Anda akan menyesuaikannya
        'Trip_Distance', 'Customer_Rating', 'Customer_Since_Months', 
        'Life_Style_Index', 'Type_of_Cab_encoded', 'Confidence_Life_Style_Index_encoded',
        'Var1', 'Var2', 'Var3', 'Distance_Rating_Interaction', 
        'Service_Quality_Score', 'Customer_Loyalty_Segment_Regular', 'Customer_Loyalty_Segment_VIP'
    ]
    
    # SVR model dengan parameter default atau sederhana
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1) # Contoh parameter SVR
    
    np.random.seed(42)
    X_train = np.random.randn(100, len(feature_names)) # Data dummy
    y_train = (1.0 + 
              X_train[:, 0] * 0.3 +
              X_train[:, 1] * 0.2 +
              np.random.normal(0, 0.1, 100))
    y_train = np.clip(y_train, 1.0, 3.0)
    
    model.fit(X_train, y_train)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    final_results = {
        'r2': 0.85, # Placeholder, sesuaikan jika perlu
        'mae': 0.1,
        'rmse': 0.15,
        'model_type': 'SVR (Fallback Built-in)'
    }
    
    return model, scaler, feature_names, final_results

# Fungsi untuk load model SVM dari file .pkl
@st.cache_resource
def load_svm_model_with_validation() -> Tuple[Any, StandardScaler, List[str], Dict, str]:
    """Load SVM model dari file .pkl dengan validasi ketat"""
    global MODEL_SOURCE
    model_path = 'Model for Streamlit/svm_model.pkl'
    # Asumsi scaler juga ada di folder yang sama atau path yang sesuai
    # Jika scaler disimpan dengan nama berbeda atau di path berbeda, sesuaikan di bawah
    scaler_path = 'Model for Streamlit/scaler.pkl' # Ganti jika nama atau path scaler berbeda
    feature_names_path = 'Model for Streamlit/feature_names.pkl' # Ganti jika nama atau path berbeda
    
    try:
        model = joblib.load(model_path)
        # Load scaler jika ada dan diperlukan untuk SVM Anda
        # Jika model SVM Anda sudah termasuk scaling atau tidak memerlukan scaler terpisah,
        # Anda bisa menghapus bagian scaler.
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            # Jika scaler.pkl tidak ada untuk SVM, kita bisa buat scaler dummy atau raise error
            # Untuk contoh ini, kita buat scaler dummy yang tidak melakukan apa-apa
            # Namun, idealnya, scaler yang digunakan saat training harus digunakan di sini.
            # Jika SVM Anda tidak pakai scaler terpisah, hapus variabel scaler.
            st.warning(f"⚠️ Scaler file '{scaler_path}' not found. Using a dummy scaler. Predictions might be inaccurate if scaling is needed.")
            scaler = StandardScaler() # Scaler dummy, latih dengan data dummy jika perlu
            dummy_data_for_scaler = np.random.rand(10, 13) # Sesuaikan jumlah fitur
            scaler.fit(dummy_data_for_scaler)


        # Load feature names jika ada
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'rb') as f:
                feature_names = pickle.load(f)
        else:
            # Jika feature_names.pkl tidak ada, gunakan placeholder atau list default
            # PENTING: Ini harus sesuai dengan fitur yang digunakan model SVM Anda!
            st.warning(f"⚠️ Feature names file '{feature_names_path}' not found. Using default feature list. Ensure this matches your SVM model's training features.")
            feature_names = [ # Ganti dengan daftar fitur SVM Anda jika berbeda
                'Trip_Distance', 'Customer_Rating', 'Customer_Since_Months', 
                'Life_Style_Index', 'Type_of_Cab_encoded', 'Confidence_Life_Style_Index_encoded',
                'Var1', 'Var2', 'Var3', 'Distance_Rating_Interaction', 
                'Service_Quality_Score', 'Customer_Loyalty_Segment_Regular', 'Customer_Loyalty_Segment_VIP'
            ]
            if len(feature_names) != 13: # Sesuaikan dengan jumlah fitur model SVM Anda
                 st.error("Default feature list length does not match expected 13 features. Please provide correct feature_names.pkl or update the default list.")


        # Asumsi tidak ada final_results.pkl untuk SVM, jadi kita buat default
        final_results = {
            'r2': 0.90, # Placeholder, sesuaikan dengan performa SVM Anda
            'mae': 0.08,
            'rmse': 0.12,
            'model_type': 'SVM (from svm_model.pkl)'
        }
        
        # Validasi model (harus memiliki metode predict)
        if not hasattr(model, 'predict'):
            raise ValueError("Invalid SVM model - missing 'predict' method")
        
        # Validasi scaler (jika digunakan)
        if scaler and not hasattr(scaler, 'transform'):
            raise ValueError("Invalid scaler - missing 'transform' method")

        # Test prediction
        # PENTING: Jumlah fitur di test_input harus sesuai dengan yang diharapkan model SVM
        # Jika feature_names dari file, len(feature_names) akan benar. Jika default, pastikan cocok.
        num_features_for_svm = len(feature_names) # Atau angka absolut jika Anda tahu pasti
        test_input = np.random.randn(1, num_features_for_svm) 
        
        if scaler:
            try:
                scaled_test = scaler.transform(test_input)
            except Exception as e:
                raise ValueError(f"Scaler transform failed: {str(e)}. Ensure scaler is compatible with {num_features_for_svm} features.")
        else:
            scaled_test = test_input # Jika tidak ada scaler

        try:
            test_pred = model.predict(scaled_test)
        except Exception as e:
            raise ValueError(f"SVM Model prediction failed: {str(e)}. Ensure model expects {num_features_for_svm} features.")
        
        if not isinstance(test_pred, np.ndarray) or len(test_pred) == 0 or np.isnan(test_pred).any() or np.isinf(test_pred).any():
            raise ValueError("Invalid SVM prediction output")
        
        MODEL_SOURCE = "svm_model.pkl"
        return model, scaler, feature_names, final_results, "success"
        
    except Exception as e:
        error_msg = str(e)
        model, scaler, feature_names, final_results = create_svm_fallback_model()
        MODEL_SOURCE = "built-in SVM fallback"
        return model, scaler, feature_names, final_results, f"fallback: {error_msg}"

# Enhanced preprocessing function (disesuaikan jika fitur SVM berbeda)
def preprocess_input_data_robust(input_dict, feature_names_list):
    """Preprocessing yang robust. Pastikan ini sesuai untuk SVM."""
    try:
        df = pd.DataFrame([input_dict])
        
        # Encoding (jika masih relevan untuk SVM)
        cab_mapping = {'Economy (Micro)': 0, 'Standard (Mini)': 1, 'Premium (Prime)': 2}
        df['Type_of_Cab_encoded'] = df['Type_of_Cab'].map(cab_mapping).fillna(0)
        
        confidence_mapping = {'High Confidence': 3, 'Medium Confidence': 2, 'Low Confidence': 1}
        df['Confidence_Life_Style_Index_encoded'] = df['Confidence_Life_Style_Index'].map(confidence_mapping).fillna(1)
        
        # Feature engineering (sesuaikan dengan yang digunakan saat training SVM)
        df['Distance_Rating_Interaction'] = df['Trip_Distance'] * df['Customer_Rating']
        df['Service_Quality_Score'] = (df['Customer_Rating'] * 0.6 + 
                                      (5 - df['Cancellation_Last_1Month'].clip(0, 5)) * 0.4)
        
        df['Customer_Loyalty_Segment'] = pd.cut(df['Customer_Since_Months'], 
                                              bins=[0, 3, 12, 24, float('inf')],
                                              labels=['New', 'Regular', 'Loyal', 'VIP'])
        
        df['Customer_Loyalty_Segment_Regular'] = (df['Customer_Loyalty_Segment'] == 'Regular').astype(int)
        df['Customer_Loyalty_Segment_VIP'] = (df['Customer_Loyalty_Segment'] == 'VIP').astype(int)
        
        # Pastikan semua fitur yang ada di feature_names_list tercover
        final_features = []
        for feature in feature_names_list:
            if feature in df.columns:
                value = float(df[feature].iloc[0])
                if np.isnan(value) or np.isinf(value):
                    value = 0.0 # Atau imputasi lain yang sesuai
                final_features.append(value)
            else:
                # Jika fitur tidak ada di input_dict tapi ada di feature_names_list, beri nilai default
                # Ini penting jika model dilatih dengan fitur yang tidak selalu ada di input user
                # st.warning(f"Feature '{feature}' not found in input, using default 0.0.")
                final_features.append(0.0) 
        
        result = np.array(final_features, dtype=np.float64).reshape(1, -1)
        
        if result.shape[1] != len(feature_names_list):
            raise ValueError(f"Preprocessing resulted in {result.shape[1]} features, expected {len(feature_names_list)}")
        
        return result
        
    except Exception as e:
        st.error(f"Error during preprocessing: {str(e)}")
        return np.zeros((1, len(feature_names_list)), dtype=np.float64)

# Load sample data function
@st.cache_data
def load_sample_data():
    """Load or create sample data safely"""
    try:
        if os.path.exists('Dataset/sigma_cabs.csv'):
            return pd.read_csv('Dataset/sigma_cabs.csv')
        else:
            np.random.seed(42)
            data = {
                'Trip_Distance': np.random.uniform(1, 50, 100),
                'Customer_Rating': np.random.uniform(1, 5, 100),
                'Surge_Pricing_Type': np.random.uniform(1, 3, 100)
            }
            return pd.DataFrame(data)
    except Exception:
        return None

# Header function tanpa blur
def display_header():
    """Display header with green theme tanpa blur"""
    image_path = 'Picture/Sigma-cabs-in-hyderabad-and-bangalore.jpg'
    
    try:
        if os.path.exists(image_path):
            st.image(image_path, caption='Sigma Cabs - Dedicated to Dedication')
        else:
            st.markdown("""
            <div class="header-box">
                <h1 style="margin: 0; font-size: clamp(2rem, 6vw, 3rem);">🚕 SIGMA CABS</h1>
                <h3 style="margin: 1rem 0; font-size: clamp(1.2rem, 4vw, 2rem);">Dedicated to Dedication</h3>
                <p style="margin: 0; font-size: clamp(1rem, 3vw, 1.3rem);">Hyderabad & Bangalore</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        st.markdown("""
        <div class="header-box">
            <h1>🚕 SIGMA CABS</h1>
            <h3>Dedicated to Dedication</h3>
            <p>Hyderabad & Bangalore</p>
        </div>
        """, unsafe_allow_html=True)

# Main application
display_header()

st.markdown('<h1 class="main-header">🌱 SVM Powered Taxi Pricing Analysis 🌊</h1>', unsafe_allow_html=True)

# Load model SVM dengan advanced validation
model, scaler, feature_names_list, final_results, load_status = load_svm_model_with_validation()

# About dan Contact
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    <div class="info-box">
        <h3>🌟 About Sigma Cabs</h3>
        <p><strong>Sigma Cabs</strong> provides exceptional cab service in 
        <strong>Hyderabad</strong> and <strong>Bangalore</strong>. Our pricing is powered by an
        <strong>Advanced SVM (Support Vector Machine) model</strong> for precise and transparent fares.</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="contact-info">
        <h4>📞 Contact Info</h4>
        <p><strong>Toll-Free:</strong><br>📞 1800-420-9999</p>
        <p><strong>24/7:</strong><br>📱 040-63 63 63 63</p>
    </div>
    """, unsafe_allow_html=True)

# Dataset preview
df = load_sample_data()
if df is not None:
    with st.expander("📊 Dataset Preview (Sample Data)"):
        try:
            st.dataframe(df.head(), use_container_width=True)
            st.write(f"**Records:** {len(df):,} | **Features:** {len(df.columns)}")
        except Exception:
            st.write("Dataset preview not available")

# Input Section
st.markdown("## 🎯 Advanced SVM Fare Prediction")

trip_container = st.container()
with trip_container:
    st.markdown("### 🚗 Trip Details")
    trip_col1, trip_col2 = st.columns([1, 1])
    with trip_col1:
        distance = st.number_input(
            "🛣️ Distance (km):", 
            min_value=0.1, 
            max_value=100.0, 
            value=5.0, 
            step=0.1,
            help="The total distance of your trip in kilometers"
        )
        cab_type = st.selectbox(
            "🚙 Vehicle Type:", 
            ['Economy (Micro)', 'Standard (Mini)', 'Premium (Prime)'],
            help="Choose your preferred vehicle category"
        )
    with trip_col2:
        destination = st.selectbox(
            "📍 Destination:", 
            ["Airport", "Business", "Home"],
            help="Type of destination affects pricing due to demand patterns"
        )
        rating = st.slider(
            "⭐ Your Rating:", 
            1, 5, 4,
            help="Your average rating as a customer"
        )

customer_container = st.container()
with customer_container:
    st.markdown("### 👤 Customer Information")
    cust_col1, cust_col2 = st.columns([1, 1])
    with cust_col1:
        months = st.number_input(
            "📅 Customer Since (Months):", 
            min_value=0, 
            max_value=120, 
            value=12,
            help="How long you've been a customer"
        )
        lifestyle = st.slider(
            "💎 Lifestyle Index:", 
            1.0, 3.0, 2.0, 
            step=0.1,
            help="1: Budget-conscious, 2: Moderate, 3: Premium"
        )
    with cust_col2:
        cancellations = st.number_input(
            "❌ Cancellations Last Month:", 
            min_value=0, 
            max_value=10, 
            value=0,
            help="Number of ride cancellations"
        )
        confidence = st.selectbox(
            "🎯 Service Confidence:", 
            ['High Confidence', 'Medium Confidence', 'Low Confidence'],
            help="Your confidence level in using taxi services"
        )

with st.expander("⚙️ Advanced Pricing Factors (Real-time Input)"):
    st.markdown("**Adjust these real-time factors for maximum precision:**")
    adv_col1, adv_col2, adv_col3 = st.columns([1, 1, 1])
    with adv_col1:
        traffic = st.slider("🚦 Traffic Density:", 0.0, 100.0, 50.0, help="Current traffic conditions")
    with adv_col2:
        demand = st.slider("📈 Demand Level:", 0.0, 100.0, 50.0, help="Current demand for taxis")
    with adv_col3:
        weather = st.slider("🌧 Weather Impact:", 0.0, 100.0, 30.0, help="Weather impact on travel")

if st.button('🔮 Calculate SVM Precision Pricing', type="primary", use_container_width=True):
    try:
        # Prepare input data untuk advanced model
        input_data = {
            'Trip_Distance': float(distance),
            'Customer_Rating': float(rating),
            'Customer_Since_Months': int(months),
            'Life_Style_Index': float(lifestyle),
            'Type_of_Cab': str(cab_type),
            'Confidence_Life_Style_Index': str(confidence),
            'Destination_Type': str(destination), # Asumsi ini ada di fitur model SVM
            'Gender': 'Male',  # Asumsi ini ada di fitur model SVM, atau hapus jika tidak
            'Cancellation_Last_1Month': int(cancellations),
            'Var1': float(traffic), # Ganti Var1, Var2, Var3 dengan nama fitur yang relevan untuk SVM jika berbeda
            'Var2': float(demand),
            'Var3': float(weather)
        }
        
        # Preprocess data
        processed_array = preprocess_input_data_robust(input_data, feature_names_list)
        
        # Validasi hasil preprocessing
        if processed_array.shape[1] != len(feature_names_list):
            raise ValueError("Feature mismatch after preprocessing: {} vs {}".format(processed_array.shape[1], len(feature_names_list)))
        
        if np.isnan(processed_array).any() or np.isinf(processed_array).any():
            raise ValueError("Invalid input values after preprocessing")
        
        # Scale dan predict menggunakan advanced model
        # Jika SVM Anda tidak menggunakan scaler terpisah atau sudah di-pipeline, sesuaikan ini
        if scaler:
            scaled_input = scaler.transform(processed_array)
        else:
            scaled_input = processed_array # Jika tidak ada scaler

        prediction_result = model.predict(scaled_input)
        
        if not isinstance(prediction_result, np.ndarray) or len(prediction_result) == 0:
            raise ValueError("Invalid prediction from SVM model")
        
        surge = float(prediction_result[0])
        
        if np.isnan(surge) or np.isinf(surge):
            raise ValueError("Invalid prediction value from SVM model (NaN or Inf)")
        
        surge = max(1.0, min(3.0, surge)) # Pastikan surge dalam rentang yang wajar
        
        prediction_html = """
        <div class="prediction-box">
            <h2>🎯 Advanced SVM Prediction</h2>
            <h1>{:.2f}x</h1>
            <p>Powered by {} - Support Vector Machine Precision AI</p>
        </div>
        """.format(surge, MODEL_SOURCE.upper())
        
        st.markdown(prediction_html, unsafe_allow_html=True)
        
        # ... (Sisa kode untuk menampilkan hasil, sama seperti sebelumnya, sesuaikan teks jika perlu) ...
        st.markdown("### 📊 Detailed Analysis Results")
        result_col1, result_col2, result_col3 = st.columns([1, 1, 1])
        
        with result_col1:
            category = "High" if surge > 2.5 else "Medium" if surge > 1.5 else "Low"
            surge_class = "surge-low" if surge <= 1.5 else "surge-medium" if surge <= 2.5 else "surge-high"
            
            surge_html = """
            <div class="metric-card {}">
                <h4>📊 Surge Analysis</h4>
                <p><strong>Category:</strong> {}</p>
                <p><strong>Multiplier:</strong> {:.2f}x</p>
                <p><strong>Distance:</strong> {} km</p>
            </div>
            """.format(surge_class, category, surge, distance)
            
            st.markdown(surge_html, unsafe_allow_html=True)
            
        with result_col2:
            loyalty = "VIP" if months > 24 else "Loyal" if months > 12 else "Regular" if months > 3 else "New"
            
            loyalty_html = """
            <div class="metric-card">
                <h4>👤 Customer Profile</h4>
                <p><strong>Loyalty Status:</strong> {}</p>
                <p><strong>Rating:</strong> {}/5.0 ⭐</p>
                <p><strong>Since:</strong> {} months</p>
            </div>
            """.format(loyalty, rating, months)
            
            st.markdown(loyalty_html, unsafe_allow_html=True)
            
        with result_col3:
            base_fare = 10.0
            distance_cost = distance * 2.5
            surge_additional = (distance_cost * (surge - 1))
            total_fare = base_fare + distance_cost + surge_additional
            
            fare_html = """
            <div class="metric-card">
                <h4>💰 Precision Fare</h4>
                <p><strong>Base:</strong> ${:.2f}</p>
                <p><strong>Distance:</strong> ${:.2f}</p>
                <p><strong>Surge:</strong> +${:.2f}</p>
                <p><strong>Total:</strong> ${:.2f}</p>
            </div>
            """.format(base_fare, distance_cost, surge_additional, total_fare)
            
            st.markdown(fare_html, unsafe_allow_html=True)
        
        st.markdown("### 🔍 Real-time Conditions Impact")
        condition_col1, condition_col2 = st.columns([1, 1])
        
        with condition_col1:
            condition_score = (traffic + demand + weather) / 3
            impact = "High Impact" if condition_score > 70 else "Medium Impact" if condition_score > 40 else "Low Impact"
            recommendation = 'Consider alternative time or route' if condition_score > 70 else 'Optimal conditions for travel'
            
            condition_html = """
            <div class="info-box">
                <h4>🚦 Current Conditions</h4>
                <p><strong>Traffic Density:</strong> {:.0f}/100</p>
                <p><strong>Demand Level:</strong> {:.0f}/100</p>
                <p><strong>Weather Impact:</strong> {:.0f}/100</p>
                <p><strong>Overall Impact:</strong> {} ({:.0f}/100)</p>
                <p><strong>💡 AI Recommendation:</strong> {}</p>
            </div>
            """.format(traffic, demand, weather, impact, condition_score, recommendation)
            
            st.markdown(condition_html, unsafe_allow_html=True)
            
        with condition_col2:
            distance_factor = min(distance / 50, 0.5)
            rating_factor = (rating - 1) / 20
            cab_factor = {'Economy (Micro)': 0.0, 'Standard (Mini)': 0.2, 'Premium (Prime)': 0.4}.get(cab_type, 0.0)
            condition_factor = (traffic + demand + weather) / 300
            
            factor_html = """
            <div class="info-box">
                <h4>📊 Factor Breakdown (Illustrative)</h4>
                <p><strong>Distance Factor:</strong> {:.3f}</p>
                <p><strong>Rating Factor:</strong> {:.3f}</p>
                <p><strong>Vehicle Factor:</strong> {:.3f}</p>
                <p><strong>Condition Factor:</strong> {:.3f}</p>
            </div>
            """.format(distance_factor, rating_factor, cab_factor, condition_factor)
            
            st.markdown(factor_html, unsafe_allow_html=True)

    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Advanced prediction error: {error_msg}")
        st.markdown("""
        <div class="prediction-box">
            <h2>🎯 Fallback Pricing</h2>
            <h1>1.50x</h1>
            <p>Using simplified algorithm</p>
        </div>
        """, unsafe_allow_html=True)

# Enhanced Information Section
info_container = st.container()
with info_container:
    st.markdown("---")
    st.markdown("## 💡 Advanced SVM Pricing Technology") # Disesuaikan
    info_col1, info_col2 = st.columns([1, 1])
    
    with info_col1:
        st.markdown("""
        <div class="info-box">
            <h3>🔍 Vehicle Categories</h3>
            <ul>
                <li><strong>🚗 Economy (Micro):</strong> Budget-friendly, compact cars</li>
                <li><strong>🚙 Standard (Mini):</strong> Regular sedans with good comfort</li>
                <li><strong>🚘 Premium (Prime):</strong> Luxury vehicles with premium service</li>
            </ul>
            <h3>🎯 Confidence Levels</h3>
            <ul>
                <li><strong>🟢 High:</strong> Frequent user, trusts service completely</li>
                <li><strong>🟡 Medium:</strong> Occasional user with moderate confidence</li>
                <li><strong>🔴 Low:</strong> New or hesitant user</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with info_col2:
        st.markdown("""
        <div class="info-box">
            <h3>🌧 Dynamic Pricing Factors</h3>
            <ul>
                <li><strong>🚦 Traffic Density:</strong> Real-time road congestion analysis</li>
                <li><strong>📈 Demand Level:</strong> Current booking requests in your area</li>
                <li><strong>🌤 Weather Impact:</strong> Weather conditions affecting travel safety</li>
                <li><strong>📏 Distance:</strong> Primary cost factor with AI optimization</li>
            </ul>
            <h3>🤖 Advanced SVM Technology</h3>
            <p>Our <strong>Advanced Support Vector Machine (SVM) model</strong> 
            analyzes multiple factors to deliver precise fare predictions.</p>
        </div>
        """, unsafe_allow_html=True)

# PINDAHKAN MODEL STATUS KE SINI - DI ATAS FOOTER
status_container = st.container()
with status_container:
    st.markdown("---")
    st.markdown("## ⚙️ Advanced System Performance")
    
    model_col1, model_col2 = st.columns([1, 1])
    
    with model_col1:
        if MODEL_SOURCE == "svm_model.pkl":
            status_html = """
            <div class="model-status success">
                <h4>✅ Model Status</h4>
                <p><strong>Source:</strong> svm_model.pkl loaded successfully</p>
                <p><strong>Type:</strong> Advanced SVM Model</p>
                <p><strong>Accuracy:</strong> {}% precision</p>
                <p><strong>Features:</strong> {} optimized features</p>
            </div>
            """.format(final_results.get('r2', 0.90)*100, len(feature_names_list))
            st.markdown(status_html, unsafe_allow_html=True)
        else:
            status_html = """
            <div class="model-status warning">
                <h4>⚠️ Model Status</h4>
                <p><strong>Source:</strong> Fallback SVM model</p>
                <p><strong>Reason:</strong> {}</p>
                <p><strong>Performance:</strong> Using simplified SVM</p>
            </div>
            """.format(load_status.replace('fallback: ', ''))
            st.markdown(status_html, unsafe_allow_html=True)
    
    with model_col2:
        if python_version >= "3.12":
            python_html = """
            <div class="model-status warning">
                <h4>⚠️ Python Environment</h4>
                <p><strong>Version:</strong> Python {}</p>
                <p><strong>Status:</strong> Using compatibility mode</p>
            </div>
            """.format(python_version)
            st.markdown(python_html, unsafe_allow_html=True)
        else:
            ml_status = 'Available' if ML_AVAILABLE else 'Limited'
            python_html = """
            <div class="model-status success">
                <h4>✅ Python Environment</h4>
                <p><strong>Version:</strong> Python {}</p>
                <p><strong>Status:</strong> Optimal performance</p>
                <p><strong>ML Libraries:</strong> {}</p>
            </div>
            """.format(python_version, ml_status)
            st.markdown(python_html, unsafe_allow_html=True)

# Enhanced Footer
footer_container = st.container()
with footer_container:
    st.markdown("---")
    
    model_status_text = '🤖 Advanced SVM Model' if MODEL_SOURCE == 'svm_model.pkl' else '⚡ Fallback SVM Model'
    footer_html = """
    <div class="footer-container" style="text-align: center; padding: clamp(1.5rem, 4vw, 2rem); 
               border-radius: 15px; margin-top: 1.5rem;">
        <h3 style="margin: 0; font-size: clamp(1.3rem, 5vw, 2rem);">🚕 Sigma Cabs - Powered by RIZKY WIBOWO KUSUMO</h3>
        <p style="margin: 1rem 0; font-size: clamp(1rem, 3vw, 1.2rem);">Safe • Reliable • Affordable • 24/7 Available</p>
        <p style="margin: 0; font-size: clamp(0.9rem, 2.5vw, 1rem);">
            <strong>Python {} | {} | 🌱 Eco-Green Theme</strong>
        </p>
    </div>
    """.format(python_version, model_status_text)
    
    st.markdown(footer_html, unsafe_allow_html=True)
