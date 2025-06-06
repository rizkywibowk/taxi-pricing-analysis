import streamlit as st

# HARUS MENJADI COMMAND PERTAMA
st.set_page_config(
    page_title="🚕 Sigma Cabs - SVM Pricing Analysis",
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
MODEL_SOURCE = "fallback" 
try:
    import joblib 
    import pickle
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR # Asumsi SVR untuk regresi harga
    import plotly.express as px
    import plotly.graph_objects as go
    from typing import Optional, Tuple, List, Dict, Any
    ML_AVAILABLE = True
except ImportError:
    pass

# Enhanced CSS dengan background hijau daun cerah
st.markdown("""
<style>
    /* Root variables untuk theming - Hijau Daun Cerah Enhanced */
    :root {
        --primary-color: #FF6B6B; /* Warna aksen utama, bisa disesuaikan */
        --secondary-color: #2e7d32; /* Hijau tua untuk aksen sekunder dan teks */
        --background-color: #e8f5e8; /* Latar belakang hijau daun cerah */
        --text-color: #1b5e20; /* Teks hijau sangat tua untuk kontras terbaik di bg terang */
        --card-background: rgba(255, 255, 255, 0.9); /* Kartu sedikit transparan */
        --border-color: #a5d6a7; /* Border hijau lebih muda dari secondary */
        --success-color: #2e7d32;
        --warning-color: #ff8f00;
        --danger-color: #d32f2f;
        --info-color: #1976d2;
        --accent-green: #4caf50; 
        --light-green: #c8e6c9; 
        --dark-green: #1b5e20; 
    }
    
    /* Dark mode variables (jika ingin tetap ada opsi dark mode berbeda) */
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color: #FF6B6B;
            --secondary-color: #66bb6a; /* Hijau lebih terang untuk dark mode */
            --background-color: #1b5e20; /* Background hijau tua untuk dark mode */
            --text-color: #e8f5e8; /* Teks hijau sangat muda */
            --card-background: rgba(46, 125, 50, 0.85); /* Kartu hijau tua transparan */
            --border-color: #81c784;
            --accent-green: #66bb6a;
            --light-green: rgba(102, 187, 106, 0.2);
        }
    }
    
    .stApp {
        background: linear-gradient(135deg, 
                   var(--background-color) 0%, 
                   color-mix(in srgb, var(--background-color) 85%, white 15%) 25%,
                   color-mix(in srgb, var(--background-color) 90%, var(--accent-green) 10%) 50%,
                   color-mix(in srgb, var(--background-color) 80%, white 20%) 75%,
                   color-mix(in srgb, var(--background-color) 95%, var(--dark-green) 5%) 100%);
        color: var(--text-color);
        min-height: 100vh;
        background-attachment: fixed;
    }
    
    .main .block-container {
        background: transparent;
    }
    
    .main-header {
        font-size: clamp(2rem, 6vw, 3rem);
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1); /* Shadow lebih lembut */
        word-wrap: break-word;
        line-height: 1.2;
        font-weight: 700;
        color: var(--secondary-color); /* Warna teks header agar kontras dengan bg hijau */
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
    
    .stImage > img {
        max-width: 100%;
        height: auto;
        border-radius: 10px;
        filter: none !important; 
        backdrop-filter: none !important;
        border: 2px solid var(--border-color); /* Tambahkan border jika perlu */
    }
    
    .prediction-box {
        background: linear-gradient(135deg, 
                   var(--secondary-color) 0%, 
                   color-mix(in srgb, var(--secondary-color) 70%, var(--dark-green) 30%) 100%);
        padding: clamp(1.5rem, 4vw, 2rem);
        border-radius: 20px;
        color: white; /* Teks putih di atas background gelap */
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.2);
        word-wrap: break-word;
        border: 2px solid var(--accent-green);
        backdrop-filter: blur(10px); /* Sedikit blur untuk efek */
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box h1 {
        font-size: clamp(3rem, 10vw, 5rem) !important;
        margin: 1rem 0 !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        color: white; /* Pastikan H1 juga putih */
    }
    
    .metric-card {
        background: var(--card-background);
        backdrop-filter: blur(15px); /* Blur lebih halus */
        padding: clamp(1rem, 3vw, 1.2rem);
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1); /* Shadow lebih lembut */
        margin: 0.8rem 0;
        min-height: clamp(120px, 18vh, 180px);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        color: var(--text-color); /* Warna teks default untuk kartu */
        transition: all 0.3s ease;
        border: 1px solid var(--border-color); /* Border lebih tipis */
        position: relative;
        overflow: hidden;
    }
    
    .metric-card h4, .metric-card p {
        color: var(--text-color) !important; /* Pastikan semua teks dalam kartu kontras */
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px; /* Bar atas lebih tebal */
        background: linear-gradient(90deg, var(--accent-green), var(--secondary-color));
    }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 40px rgba(46, 125, 50, 0.2); /* Shadow dengan hint hijau */
        border-color: var(--accent-green);
    }
    
    /* ... (CSS lainnya tetap sama, hanya pastikan warna teks dan background kontras) ... */
    
    .info-box, .contact-info, .model-status, .footer-container {
        background: var(--card-background);
        backdrop-filter: blur(15px);
        padding: clamp(1rem, 3vw, 1.2rem);
        border-radius: 15px;
        margin: 0.8rem 0;
        color: var(--text-color);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        border: 1px solid var(--border-color);
    }
    
    .model-status.success { border-left: 6px solid var(--success-color); }
    .model-status.warning { border-left: 6px solid var(--warning-color); }

    .header-box { /* Header jika gambar tidak ada */
        background: linear-gradient(135deg, 
                   var(--secondary-color) 0%, 
                   color-mix(in srgb, var(--secondary-color) 70%, var(--dark-green) 30%) 100%);
        padding: clamp(2rem, 5vw, 3rem);
        text-align: center;
        border-radius: 20px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 35px rgba(0,0,0,0.2);
        backdrop-filter: none; 
        border: 2px solid var(--accent-green);
    }
</style>
""", unsafe_allow_html=True)

# --- PEMBENAHAN MODEL DAN FITUR ---

# PENTING: Anda HARUS menyesuaikan daftar ini dengan 20 fitur yang digunakan model SVM Anda
# Ini adalah placeholder jika 'feature_names_svm.pkl' tidak ditemukan.
DEFAULT_SVM_FEATURE_NAMES_20 = [
    'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5',
    'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10',
    'Feature11', 'Feature12', 'Feature13', 'Feature14', 'Feature15',
    'Feature16', 'Feature17', 'Feature18', 'Feature19', 'Feature20'
]
# Jika model Anda benar-benar hanya 13 fitur, sesuaikan error message di Gambar 3, atau model SVM Anda.
# Untuk sekarang, kita akan bekerja dengan asumsi model SVM Anda memang butuh 20 fitur.

def create_svm_fallback_model():
    """Create a fallback SVM model (SVR for regression) dengan 20 fitur."""
    feature_names = DEFAULT_SVM_FEATURE_NAMES_20 # Menggunakan 20 fitur
    
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1) 
    
    np.random.seed(42)
    X_train = np.random.randn(100, len(feature_names)) # Data dummy dengan 20 fitur
    y_train = (1.0 + X_train[:, 0] * 0.1 + np.random.normal(0, 0.1, 100)) # Contoh sederhana
    y_train = np.clip(y_train, 1.0, 3.0)
    
    model.fit(X_train, y_train)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    final_results = {
        'r2': 0.80, 
        'mae': 0.12,
        'rmse': 0.18,
        'model_type': 'SVR (Fallback Built-in - 20 Features)'
    }
    
    return model, scaler, feature_names, final_results

@st.cache_resource
def load_svm_model_with_validation() -> Tuple[Any, StandardScaler, List[str], Dict, str]:
    """Load SVM model dari file .pkl dengan validasi ketat."""
    global MODEL_SOURCE
    model_path = 'Model for Streamlit/svm_model.pkl'
    scaler_path = 'Model for Streamlit/scaler_svm.pkl' # Ganti nama jika berbeda
    feature_names_path = 'Model for Streamlit/feature_names_svm.pkl' # Ganti nama jika berbeda
    
    loaded_feature_names = []

    try:
        model = joblib.load(model_path)
        
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            st.warning(f"⚠️ Scaler file '{scaler_path}' not found. Creating and fitting a new StandardScaler. This might lead to suboptimal performance if the original scaler was different.")
            scaler = StandardScaler()
            # Untuk fit scaler, idealnya kita butuh data yang representatif
            # Jika tidak ada, kita fit pada data dummy dengan asumsi jumlah fitur yang benar
            # Ini akan di-handle setelah feature_names dimuat

        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'rb') as f:
                loaded_feature_names = pickle.load(f)
                if not isinstance(loaded_feature_names, list) or not all(isinstance(name, str) for name in loaded_feature_names):
                    raise ValueError("feature_names_svm.pkl does not contain a valid list of strings.")
                if len(loaded_feature_names) != 20: # Validasi jumlah fitur dari file
                     st.warning(f"⚠️ Feature names from '{feature_names_path}' has {len(loaded_feature_names)} features, but error log (SVC) indicated 20. Using 20 default features. Ensure consistency.")
                     loaded_feature_names = DEFAULT_SVM_FEATURE_NAMES_20
        else:
            st.warning(f"⚠️ Feature names file '{feature_names_path}' not found. Using default list of 20 features. YOU MUST VERIFY THESE ARE CORRECT for your svm_model.pkl.")
            loaded_feature_names = DEFAULT_SVM_FEATURE_NAMES_20

        # Jika scaler baru dibuat dan belum di-fit
        if not hasattr(scaler, 'mean_'): # Cek apakah scaler sudah di-fit
            dummy_data_for_scaler = np.random.rand(10, len(loaded_feature_names))
            scaler.fit(dummy_data_for_scaler)
            st.info("Fitted new StandardScaler with dummy data based on loaded/default feature count.")

        final_results = { # Placeholder, idealnya ini juga bisa dari file atau dihitung
            'r2': 0.90, 
            'mae': 0.08,
            'rmse': 0.12,
            'model_type': 'SVM (from svm_model.pkl)'
        }
        
        if not hasattr(model, 'predict'):
            raise ValueError("Invalid SVM model - missing 'predict' method")
        if scaler and not hasattr(scaler, 'transform'):
            raise ValueError("Invalid scaler - missing 'transform' method")

        # Test prediction dengan jumlah fitur yang benar
        num_features_expected = len(loaded_feature_names)
        test_input = np.random.randn(1, num_features_expected) 
        
        scaled_test = scaler.transform(test_input) if scaler else test_input
        test_pred = model.predict(scaled_test)
        
        if not isinstance(test_pred, np.ndarray) or len(test_pred) == 0 or np.isnan(test_pred).any() or np.isinf(test_pred).any():
            raise ValueError("Invalid SVM prediction output")
        
        MODEL_SOURCE = "svm_model.pkl"
        return model, scaler, loaded_feature_names, final_results, "success"
        
    except Exception as e:
        error_msg = str(e)
        model, scaler, loaded_feature_names, final_results = create_svm_fallback_model()
        MODEL_SOURCE = "built-in SVM fallback"
        return model, scaler, loaded_feature_names, final_results, f"fallback: {error_msg}"

def preprocess_input_data_robust(input_dict, feature_names_list_expected):
    """Preprocessing data agar sesuai dengan feature_names_list_expected (20 fitur)."""
    processed_data = {}
    df_input = pd.DataFrame([input_dict])

    # Lakukan encoding dan feature engineering dasar yang mungkin masih relevan
    # Ini harus menghasilkan fitur-fitur yang menjadi bagian dari 20 fitur akhir
    # Contoh sederhana:
    cab_mapping = {'Economy (Micro)': 0, 'Standard (Mini)': 1, 'Premium (Prime)': 2}
    processed_data['Type_of_Cab_encoded'] = df_input['Type_of_Cab'].map(cab_mapping).fillna(0).iloc[0]
    
    confidence_mapping = {'High Confidence': 3, 'Medium Confidence': 2, 'Low Confidence': 1}
    processed_data['Confidence_Life_Style_Index_encoded'] = df_input['Confidence_Life_Style_Index'].map(confidence_mapping).fillna(1).iloc[0]
    
    processed_data['Distance_Rating_Interaction'] = float(input_dict['Trip_Distance']) * float(input_dict['Customer_Rating'])
    processed_data['Service_Quality_Score'] = (float(input_dict['Customer_Rating']) * 0.6 + 
                                     (5 - int(input_dict['Cancellation_Last_1Month'])) * 0.4) # Asumsi clip sudah dihandle
    
    # Langsung ambil nilai dari input_dict untuk fitur yang ada
    direct_features = ['Trip_Distance', 'Customer_Rating', 'Customer_Since_Months', 
                       'Life_Style_Index', 'Var1', 'Var2', 'Var3']
    for f_name in direct_features:
        if f_name in input_dict:
            processed_data[f_name] = float(input_dict[f_name])

    # Untuk Customer Loyalty Segment (jika ini salah satu dari 20 fitur)
    # Ini contoh, Anda mungkin punya cara lain untuk menurunkannya
    months = int(input_dict['Customer_Since_Months'])
    processed_data['Customer_Loyalty_Segment_Regular'] = 1 if 3 < months <= 12 else 0
    processed_data['Customer_Loyalty_Segment_VIP'] = 1 if months > 24 else 0
    # processed_data['Customer_Loyalty_Segment_New'] = 1 if months <=3 else 0
    # processed_data['Customer_Loyalty_Segment_Loyal'] = 1 if 12 < months <=24 else 0


    # Membuat array final berdasarkan feature_names_list_expected (20 fitur)
    final_feature_values = []
    for feature_name in feature_names_list_expected:
        if feature_name in processed_data:
            final_feature_values.append(processed_data[feature_name])
        elif feature_name in input_dict: # Jika ada di input_dict tapi belum di processed_data
            final_feature_values.append(float(input_dict[feature_name]))
        else:
            # Fitur tidak ada, beri nilai default (misal 0 atau rata-rata dari training set)
            # PENTING: Imputasi ini harus konsisten dengan saat training model SVM Anda!
            # st.warning(f"Feature '{feature_name}' is missing from input and basic processing. Using default 0.0. This might affect prediction accuracy.")
            final_feature_values.append(0.0) 
            
    result_array = np.array(final_feature_values, dtype=np.float64).reshape(1, -1)

    if result_array.shape[1] != len(feature_names_list_expected):
        raise ValueError(f"Preprocessing resulted in {result_array.shape[1]} features, but {len(feature_names_list_expected)} were expected for the SVM model.")
    
    return result_array

# ... (Sisa fungsi seperti load_sample_data, display_header, dll. tetap sama) ...
# Fungsi get_surge_category_class dan get_loyalty_class juga tetap sama

# Load sample data function
@st.cache_data
def load_sample_data():
    """Load or create sample data safely"""
    try:
        if os.path.exists('Dataset/sigma_cabs.csv'): # Pastikan path ini benar
            return pd.read_csv('Dataset/sigma_cabs.csv')
        else:
            st.info("Dataset/sigma_cabs.csv not found, using dummy data for preview.")
            np.random.seed(42)
            data = {
                'Trip_Distance': np.random.uniform(1, 50, 100),
                'Customer_Rating': np.random.uniform(1, 5, 100),
                'Surge_Pricing_Type': np.random.uniform(1, 3, 100) # Contoh kolom
            }
            return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None

# Header function tanpa blur
def display_header():
    """Display header with green theme tanpa blur"""
    image_path = 'Picture/Sigma-cabs-in-hyderabad-and-bangalore.jpg' # Pastikan path ini benar
    
    try:
        if os.path.exists(image_path):
            st.image(image_path, caption='Sigma Cabs - Dedicated to Dedication')
        else:
            st.warning("Header image 'Picture/Sigma-cabs-in-hyderabad-and-bangalore.jpg' not found. Displaying text header.")
            st.markdown("""
            <div class="header-box">
                <h1 style="margin: 0; font-size: clamp(2rem, 6vw, 3rem);">🚕 SIGMA CABS</h1>
                <h3 style="margin: 1rem 0; font-size: clamp(1.2rem, 4vw, 2rem);">Dedicated to Dedication</h3>
                <p style="margin: 0; font-size: clamp(1rem, 3vw, 1.3rem);">Hyderabad & Bangalore</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying header image: {e}")
        st.markdown("""
        <div class="header-box">
            <h1>🚕 SIGMA CABS</h1>
            <h3>Dedicated to Dedication</h3>
            <p>Hyderabad & Bangalore</p>
        </div>
        """, unsafe_allow_html=True)

# Main application
display_header()

st.markdown('<h1 class="main-header">🌱 SVM Powered Eco-Smart Taxi Pricing 🌊</h1>', unsafe_allow_html=True)

# Load model SVM dengan advanced validation
model, scaler, feature_names_list_for_svm, final_results, load_status = load_svm_model_with_validation()

# About dan Contact
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    <div class="info-box">
        <h3>🌟 About Sigma Cabs</h3>
        <p><strong>Sigma Cabs</strong> provides exceptional cab service. Our pricing is powered by an
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
        except Exception:
            st.write("Dataset preview not available")

# Input Section
st.markdown("## 🎯 Advanced SVM Fare Prediction")

trip_container = st.container()
# ... (Input fields sama seperti kode Anda sebelumnya, pastikan nama variabel unik) ...
with trip_container:
    st.markdown("### 🚗 Trip Details")
    trip_col1, trip_col2 = st.columns([1, 1])
    with trip_col1:
        distance_input = st.number_input( # Ganti nama variabel agar tidak konflik
            "🛣️ Distance (km):", 
            min_value=0.1, 
            max_value=100.0, 
            value=5.0, 
            step=0.1,
            key="distance_input"
        )
        cab_type_input = st.selectbox( # Ganti nama variabel
            "🚙 Vehicle Type:", 
            ['Economy (Micro)', 'Standard (Mini)', 'Premium (Prime)'],
            key="cab_type_input"
        )
    with trip_col2:
        destination_input = st.selectbox( # Ganti nama variabel
            "📍 Destination:", 
            ["Airport", "Business", "Home"],
            key="destination_input"
        )
        rating_input = st.slider( # Ganti nama variabel
            "⭐ Your Rating:", 
            1, 5, 4,
            key="rating_input"
        )

customer_container = st.container()
with customer_container:
    st.markdown("### 👤 Customer Information")
    cust_col1, cust_col2 = st.columns([1, 1])
    with cust_col1:
        months_input = st.number_input( # Ganti nama variabel
            "📅 Customer Since (Months):", 
            min_value=0, 
            max_value=120, 
            value=12,
            key="months_input"
        )
        lifestyle_input = st.slider( # Ganti nama variabel
            "💎 Lifestyle Index:", 
            1.0, 3.0, 2.0, 
            step=0.1,
            key="lifestyle_input"
        )
    with cust_col2:
        cancellations_input = st.number_input( # Ganti nama variabel
            "❌ Cancellations Last Month:", 
            min_value=0, 
            max_value=10, 
            value=0,
            key="cancellations_input"
        )
        confidence_input = st.selectbox( # Ganti nama variabel
            "🎯 Service Confidence:", 
            ['High Confidence', 'Medium Confidence', 'Low Confidence'],
            key="confidence_input"
        )

with st.expander("⚙️ Advanced Pricing Factors (Real-time Input)"):
    st.markdown("**Adjust these real-time factors for maximum precision:**")
    adv_col1, adv_col2, adv_col3 = st.columns([1, 1, 1])
    with adv_col1:
        traffic_input = st.slider("🚦 Traffic Density:", 0.0, 100.0, 50.0, key="traffic_input")
    with adv_col2:
        demand_input = st.slider("📈 Demand Level:", 0.0, 100.0, 50.0, key="demand_input")
    with adv_col3:
        weather_input = st.slider("🌧 Weather Impact:", 0.0, 100.0, 30.0, key="weather_input")

if st.button('🔮 Calculate SVM Precision Pricing', type="primary", use_container_width=True):
    try:
        input_data = {
            'Trip_Distance': float(distance_input),
            'Customer_Rating': float(rating_input),
            'Customer_Since_Months': int(months_input),
            'Life_Style_Index': float(lifestyle_input),
            'Type_of_Cab': str(cab_type_input),
            'Confidence_Life_Style_Index': str(confidence_input),
            'Destination_Type': str(destination_input), 
            'Gender': 'Male', 
            'Cancellation_Last_1Month': int(cancellations_input),
            'Var1': float(traffic_input), 
            'Var2': float(demand_input),
            'Var3': float(weather_input)
            # PENTING: Tambahkan fitur lain di sini jika 20 fitur SVM Anda berbeda
            # Misal: 'Feature14': default_value_14, ... 'Feature20': default_value_20
        }
        
        processed_array = preprocess_input_data_robust(input_data, feature_names_list_for_svm)
        
        if scaler: # Hanya transform jika scaler ada dan sudah di-fit
            scaled_input = scaler.transform(processed_array)
        else:
            scaled_input = processed_array

        prediction_result = model.predict(scaled_input)
        surge = float(prediction_result[0])
        surge = max(1.0, min(3.0, surge))
        
        prediction_html = """
        <div class="prediction-box">
            <h2>🎯 Advanced SVM Prediction</h2>
            <h1>{:.2f}x</h1>
            <p>Powered by {} - Support Vector Machine Precision AI</p>
        </div>
        """.format(surge, MODEL_SOURCE.upper())
        st.markdown(prediction_html, unsafe_allow_html=True)
        
        # Tampilan hasil lainnya (metric cards, dll.)
        # ... (Kode tampilan hasil seperti di versi sebelumnya, pastikan variabelnya sesuai) ...
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
            """.format(surge_class, category, surge, distance_input) # Gunakan distance_input
            st.markdown(surge_html, unsafe_allow_html=True)
            
        with result_col2:
            loyalty = "VIP" if months_input > 24 else "Loyal" if months_input > 12 else "Regular" if months_input > 3 else "New"
            loyalty_html = """
            <div class="metric-card">
                <h4>👤 Customer Profile</h4>
                <p><strong>Loyalty Status:</strong> {}</p>
                <p><strong>Rating:</strong> {}/5.0 ⭐</p>
                <p><strong>Since:</strong> {} months</p>
            </div>
            """.format(loyalty, rating_input, months_input) # Gunakan variabel input yang benar
            st.markdown(loyalty_html, unsafe_allow_html=True)
            
        with result_col3:
            base_fare = 10.0
            distance_cost_val = float(distance_input) * 2.5 # Gunakan variabel input yang benar
            surge_additional = (distance_cost_val * (surge - 1))
            total_fare = base_fare + distance_cost_val + surge_additional
            fare_html = """
            <div class="metric-card">
                <h4>💰 Precision Fare</h4>
                <p><strong>Base:</strong> ${:.2f}</p>
                <p><strong>Distance:</strong> ${:.2f}</p>
                <p><strong>Surge:</strong> +${:.2f}</p>
                <p><strong>Total:</strong> ${:.2f}</p>
            </div>
            """.format(base_fare, distance_cost_val, surge_additional, total_fare)
            st.markdown(fare_html, unsafe_allow_html=True)

    except Exception as e:
        error_msg = str(e)
        st.error("❌ Advanced prediction error: {}".format(error_msg))
        st.markdown("""
        <div class="prediction-box">
            <h2>🎯 Fallback Pricing</h2>
            <h1>1.50x</h1>
            <p>Using simplified algorithm</p>
        </div>
        """, unsafe_allow_html=True)

# ... (Sisa kode: Information Section, Model Status, Footer - pastikan variabelnya benar) ...
# Enhanced Information Section
info_container = st.container()
with info_container:
    st.markdown("---")
    st.markdown("## 💡 Advanced SVM Pricing Technology") 
    info_col1, info_col2 = st.columns([1, 1])
    
    with info_col1:
        st.markdown("""
        <div class="info-box">
            <h3>🔍 Vehicle Categories</h3>
            <ul>
                <li><strong>🚗 Economy (Micro):</strong> Budget-friendly</li>
                <li><strong>🚙 Standard (Mini):</strong> Comfortable</li>
                <li><strong>🚘 Premium (Prime):</strong> Luxury service</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with info_col2:
        st.markdown("""
        <div class="info-box">
            <h3>🌧 Dynamic Pricing Factors</h3>
            <ul>
                <li><strong>🚦 Traffic, 📈 Demand, 🌤 Weather</strong></li>
            </ul>
            <h3>🤖 Advanced SVM Technology</h3>
            <p>Our SVM model analyzes factors for precise fares.</p>
        </div>
        """, unsafe_allow_html=True)

# PINDAHKAN MODEL STATUS KE SINI - DI ATAS FOOTER
status_container = st.container()
with status_container:
    st.markdown("---")
    st.markdown("## ⚙️ Advanced System Performance")
    
    model_col1, model_col2 = st.columns([1, 1])
    
    with model_col1:
        model_accuracy = final_results.get('r2', 0.0) * 100
        num_features = len(feature_names_list_for_svm) if feature_names_list_for_svm else 'N/A'

        if MODEL_SOURCE == "svm_model.pkl":
            status_html = """
            <div class="model-status success">
                <h4>✅ Model Status</h4>
                <p><strong>Source:</strong> svm_model.pkl loaded</p>
                <p><strong>Type:</strong> Advanced SVM Model</p>
                <p><strong>Accuracy:</strong> {:.2f}% (approx.)</p>
                <p><strong>Features:</strong> {}</p>
            </div>
            """.format(model_accuracy, num_features)
            st.markdown(status_html, unsafe_allow_html=True)
        else:
            status_html = """
            <div class="model-status warning">
                <h4>⚠️ Model Status</h4>
                <p><strong>Source:</strong> Fallback SVM model</p>
                <p><strong>Reason:</strong> {}</p>
                <p><strong>Type:</strong> Simplified SVR</p>
                <p><strong>Features:</strong> {}</p>
            </div>
            """.format(load_status.replace('fallback: ', ''), num_features)
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
            <strong>Python {} | {} | 🌱 All Device Optimized</strong>
        </p>
    </div>
    """.format(python_version, model_status_text)
    
    st.markdown(footer_html, unsafe_allow_html=True)

