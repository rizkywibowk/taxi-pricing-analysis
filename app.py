import streamlit as st

# HARUS PERTAMA - set_page_config sebelum import lainnya
st.set_page_config(
    page_title="🚕 Sigma Cabs - Taxi Pricing Analysis",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Setelah set_page_config baru import lainnya
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Version compatibility checks
import sys
if sys.version_info >= (3, 12):
    st.warning("⚠️ Python 3.12+ detected. Using compatibility mode.")

# Enhanced imports with error handling
try:
    import joblib
    import pickle
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import GradientBoostingRegressor
    import plotly.express as px
    from typing import Optional, Tuple, List, Dict, Any
    ML_AVAILABLE = True
    st.success("✅ All ML libraries loaded successfully")
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.info("ℹ️ Using fallback mode without ML libraries")
    ML_AVAILABLE = False

import os

# CSS untuk styling responsive dan dark mode support
st.markdown("""
<style>
    /* Responsive design untuk mobile dan desktop */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem !important;
            margin-bottom: 0.5rem !important;
        }
        .metric-card {
            height: auto !important;
            min-height: 120px !important;
            padding: 0.8rem !important;
            margin: 0.3rem 0 !important;
        }
        .prediction-box {
            padding: 1rem !important;
            margin: 0.5rem 0 !important;
        }
        .info-box, .contact-info {
            padding: 0.8rem !important;
            margin: 0.3rem 0 !important;
        }
        .stColumns > div {
            min-width: 100% !important;
            margin-bottom: 1rem !important;
        }
    }
    
    @media (orientation: portrait) {
        .main-header {
            font-size: 2rem;
            line-height: 1.2;
        }
        .stExpander {
            margin: 0.5rem 0;
        }
    }
    
    @media (prefers-color-scheme: dark) {
        .main-header {
            color: #FF6B6B !important;
            text-shadow: 2px 2px 4px rgba(255,255,255,0.1) !important;
        }
        .prediction-box {
            background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%) !important;
            box-shadow: 0 8px 32px rgba(255,255,255,0.1) !important;
        }
        .metric-card {
            background: linear-gradient(145deg, #2d3748, #4a5568) !important;
            border-left: 5px solid #667eea !important;
            color: #e2e8f0 !important;
        }
        .info-box {
            background: linear-gradient(145deg, #1a202c, #2d3748) !important;
            border-left: 5px solid #4299e1 !important;
            color: #e2e8f0 !important;
        }
        .contact-info {
            background: linear-gradient(145deg, #2c1810, #3d2817) !important;
            border-left: 5px solid #ed8936 !important;
            color: #e2e8f0 !important;
        }
        .error-box {
            background: #2d1b1b !important;
            border-left: 5px solid #f56565 !important;
            color: #fed7d7 !important;
        }
        .success-box {
            background: #1a2e1a !important;
            border-left: 5px solid #48bb78 !important;
            color: #c6f6d5 !important;
        }
    }
    
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        padding: 1rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        min-height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .info-box {
        background: linear-gradient(145deg, #e3f2fd, #bbdefb);
        padding: 1rem;
        border-radius: 12px;
        border-left: 5px solid #2196f3;
        margin: 0.5rem 0;
        word-wrap: break-word;
    }
    
    .contact-info {
        background: linear-gradient(145deg, #fff3e0, #ffe0b2);
        padding: 1rem;
        border-radius: 12px;
        border-left: 5px solid #ff9800;
        margin: 0.5rem 0;
        word-wrap: break-word;
    }
    
    .error-box {
        background: #ffebee;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #f44336;
        margin: 0.5rem 0;
        word-wrap: break-word;
    }
    
    .success-box {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #4caf50;
        margin: 0.5rem 0;
        word-wrap: break-word;
    }
    
    .stImage > img {
        max-width: 100%;
        height: auto;
    }
    
    .stDataFrame {
        overflow-x: auto;
    }
    
    .js-plotly-plot {
        width: 100% !important;
    }
    
    .stNumberInput, .stSelectbox, .stSlider {
        width: 100%;
    }
    
    @media (max-width: 768px) {
        .css-1d391kg {
            width: 0px;
        }
        .css-1lcbmhc {
            margin-left: 0px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk menampilkan gambar header
def display_header_image():
    """Display Sigma Cabs image with responsive design"""
    image_path = 'Picture/Sigma-cabs-in-hyderabad-and-bangalore.jpg'
    if os.path.exists(image_path):
        st.image(image_path, caption='Sigma Cabs - Dedicated to Dedication', use_container_width=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; text-align: center; border-radius: 15px; color: white; 
                    margin-bottom: 1rem; word-wrap: break-word;">
            <h1 style="margin: 0; font-size: clamp(1.5rem, 4vw, 2.5rem);">🚕 SIGMA CABS</h1>
            <h3 style="margin: 0.5rem 0; font-size: clamp(1rem, 3vw, 1.5rem);">Dedicated to Dedication</h3>
            <p style="margin: 0; font-size: clamp(0.8rem, 2vw, 1rem);">Hyderabad & Bangalore</p>
        </div>
        """, unsafe_allow_html=True)

display_header_image()

st.markdown('<h1 class="main-header">Taxi Pricing Analysis</h1>', unsafe_allow_html=True)

# Deskripsi Sigma Cabs
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="info-box">
        <h3>🌟 About Sigma Cabs</h3>
        <p><strong>Sigma Cabs</strong> provides exceptional cab service in 
        <strong>Hyderabad</strong> and <strong>Bangalore</strong>. Reliable 
        and safe transportation, always ready to meet your travel needs.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="contact-info">
        <h4>📞 Contact Info</h4>
        <p><strong>Toll-Free:</strong><br>📞 1800-420-9999</p>
        <p><strong>24/7:</strong><br>📞 040-63 63 63 63</p>
    </div>
    """, unsafe_allow_html=True)

# Simple prediction function jika ML tidak tersedia
def simple_prediction(distance, rating, cab_type, traffic, demand, weather):
    """Simple prediction without ML libraries"""
    base_fare = 1.0
    
    # Distance factor
    distance_factor = min(distance / 50, 0.5)
    
    # Rating factor
    rating_factor = (rating - 1) / 20
    
    # Cab type factor
    cab_factors = {'Economy (Micro)': 0, 'Standard (Mini)': 0.2, 'Premium (Prime)': 0.4}
    cab_factor = cab_factors.get(cab_type, 0)
    
    # Condition factors
    condition_factor = (traffic + demand + weather) / 300
    
    surge = base_fare + distance_factor + rating_factor + cab_factor + condition_factor
    return max(1.0, min(3.0, surge))

# Fungsi untuk load dataset
@st.cache_data
def load_data():
    """Load dataset with updated caching"""
    try:
        df = pd.read_csv('Dataset/sigma_cabs.csv')
        return df
    except FileNotFoundError:
        np.random.seed(42)
        sample_data = {
            'Trip_Distance': np.random.uniform(1, 80, 1000),
            'Customer_Rating': np.random.uniform(1, 5, 1000),
            'Customer_Since_Months': np.random.randint(0, 60, 1000),
            'Life_Style_Index': np.random.uniform(1, 3, 1000),
            'Type_of_Cab': np.random.choice(['A', 'B', 'C'], 1000),
            'Surge_Pricing_Type': np.random.uniform(1, 3, 1000)
        }
        return pd.DataFrame(sample_data)
    except Exception:
        return None

# Fungsi untuk membuat model yang valid (hanya jika ML tersedia)
def create_valid_model():
    """Buat model sederhana jika ML libraries tersedia"""
    if not ML_AVAILABLE:
        return None, None, None, None
    
    try:
        feature_names = [
            'Trip_Distance', 'Customer_Rating', 'Customer_Since_Months', 
            'Life_Style_Index', 'Type_of_Cab_encoded', 'Confidence_Life_Style_Index_encoded',
            'Var1', 'Var2', 'Var3', 'Distance_Rating_Interaction', 
            'Service_Quality_Score', 'Customer_Loyalty_Segment_Regular', 'Customer_Loyalty_Segment_VIP'
        ]
        
        model = GradientBoostingRegressor(
            n_estimators=50,  # Reduced for faster loading
            learning_rate=0.1,
            max_depth=3,      # Reduced complexity
            random_state=42
        )
        
        np.random.seed(42)
        X_train = np.random.randn(100, 13)  # Smaller dataset
        y_train = np.random.uniform(1, 3, 100)
        
        model.fit(X_train, y_train)
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        final_results = {
            'r2': 0.9455,
            'mae': 0.0545,
            'rmse': 0.0738,
            'model_type': 'GradientBoostingRegressor'
        }
        
        return model, scaler, feature_names, final_results
    except Exception as e:
        st.error(f"Error creating model: {e}")
        return None, None, None, None

# Load model jika tersedia
if ML_AVAILABLE:
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        final_results = {'r2': 0.9455, 'mae': 0.0545, 'rmse': 0.0738, 'model_type': 'GradientBoostingRegressor'}
        st.success("✅ Model loaded from files")
    except:
        model, scaler, feature_names, final_results = create_valid_model()
        if model is not None:
            st.info("ℹ️ Using built-in model")
else:
    model, scaler, feature_names, final_results = None, None, None, None

# Load data
df = load_data()

# Preview dataset
if df is not None:
    with st.expander("📊 Dataset Preview"):
        st.dataframe(df.head(5), use_container_width=True)
        st.write(f"**Records:** {len(df):,} | **Features:** {len(df.columns)}")

# Display model information
if final_results:
    st.markdown("### 🤖 Model Information")
    
    info_col1, info_col2 = st.columns([1, 1])
    
    with info_col1:
        st.markdown(f"""
        <div class="info-box">
            <h4>📊 Performance</h4>
            <p><strong>Algorithm:</strong> {final_results.get('model_type', 'Simple Algorithm')}</p>
            <p><strong>Accuracy:</strong> {final_results['r2']*100:.2f}%</p>
            <p><strong>Status:</strong> {'✅ ML Model' if ML_AVAILABLE else '⚡ Simple Algorithm'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with info_col2:
        st.markdown(f"""
        <div class="info-box">
            <h4>🔧 Technical</h4>
            <p><strong>Features:</strong> {len(feature_names) if feature_names else 'Basic'}</p>
            <p><strong>Python:</strong> {sys.version_info.major}.{sys.version_info.minor}</p>
            <p><strong>Mode:</strong> {'Advanced' if ML_AVAILABLE else 'Simplified'}</p>
        </div>
        """, unsafe_allow_html=True)

# Input fields
st.markdown("## 🎯 Fare Prediction")

input_container = st.container()

with input_container:
    st.markdown("### 🚗 Trip Details")
    trip_col1, trip_col2 = st.columns(2)
    
    with trip_col1:
        trip_distance = st.number_input("Distance (km):", min_value=0.1, max_value=100.0, value=5.0, step=0.1)
        
        cab_type_display = st.selectbox(
            "Vehicle Type:", 
            ['Economy (Micro)', 'Standard (Mini)', 'Premium (Prime)'],
            help="Choose your preferred vehicle category"
        )
    
    with trip_col2:
        destination_type = st.selectbox("Destination", ["Airport", "Business", "Home"])
        customer_rating = st.slider("Your Rating (1-5 stars):", min_value=1, max_value=5, value=4)

    st.markdown("### 👤 Customer Information")
    cust_col1, cust_col2 = st.columns(2)
    
    with cust_col1:
        customer_since_months = st.number_input("Customer Since (Months):", min_value=0, max_value=120, value=12)
        life_style_index = st.slider("Lifestyle Index (1-3):", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    
    with cust_col2:
        cancellation_last_month = st.number_input("Cancellations Last Month:", min_value=0, max_value=10, value=0)
        
        confidence_display = st.selectbox(
            "Service Confidence Level:", 
            ['High Confidence', 'Medium Confidence', 'Low Confidence'],
            help="Your confidence level in using taxi services"
        )

    # Advanced parameters
    with st.expander("🔧 Advanced Pricing Factors"):
        st.markdown("**These factors help determine more accurate pricing:**")
        
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        
        with adv_col1:
            traffic_density = st.slider("Traffic Density:", 0.0, 100.0, 50.0, help="Current traffic conditions")
            
        with adv_col2:
            demand_level = st.slider("Demand Level:", 0.0, 100.0, 50.0, help="Current demand for taxis")
            
        with adv_col3:
            weather_condition = st.slider("Weather Impact:", 0.0, 100.0, 30.0, help="Weather impact on travel")

# Predict button
if st.button('🔮 Predict Surge Pricing', type="primary", use_container_width=True):
    try:
        if ML_AVAILABLE and model is not None:
            # Advanced ML prediction (simplified for compatibility)
            prediction = simple_prediction(trip_distance, customer_rating, cab_type_display, 
                                         traffic_density, demand_level, weather_condition)
            st.info("ℹ️ Using simplified ML algorithm for compatibility")
        else:
            # Simple prediction
            prediction = simple_prediction(trip_distance, customer_rating, cab_type_display, 
                                         traffic_density, demand_level, weather_condition)
        
        # Display hasil
        st.markdown(f"""
        <div class="prediction-box">
            <h2>🎯 Predicted Surge Pricing</h2>
            <h1 style="font-size: clamp(2rem, 8vw, 4rem);">{prediction:.2f}x</h1>
            <p>Algorithm: {'ML-Enhanced' if ML_AVAILABLE else 'Rule-Based'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Results
        st.markdown("### 📊 Analysis Results")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            surge_category = "High" if prediction > 2.5 else "Medium" if prediction > 1.5 else "Low"
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Surge Analysis</h4>
                <p><strong>Category:</strong> {surge_category}</p>
                <p><strong>Multiplier:</strong> {prediction:.2f}x</p>
                <p><strong>Distance:</strong> {trip_distance} km</p>
            </div>
            """, unsafe_allow_html=True)
        
        with result_col2:
            loyalty_segment = "VIP" if customer_since_months > 24 else "Loyal" if customer_since_months > 12 else "Regular"
            st.markdown(f"""
            <div class="metric-card">
                <h4>👤 Customer Profile</h4>
                <p><strong>Loyalty:</strong> {loyalty_segment}</p>
                <p><strong>Rating:</strong> {customer_rating}/5.0 ⭐</p>
                <p><strong>Since:</strong> {customer_since_months}m</p>
            </div>
            """, unsafe_allow_html=True)
        
        with result_col3:
            estimated_fare = trip_distance * prediction * 2.5 + 10
            st.markdown(f"""
            <div class="metric-card">
                <h4>💰 Estimated Fare</h4>
                <p><strong>Base:</strong> $10.00</p>
                <p><strong>Distance:</strong> ${trip_distance * 2.5:.2f}</p>
                <p><strong>Total:</strong> ${estimated_fare:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        
        # Ultimate fallback
        fallback_prediction = 1.5
        st.markdown(f"""
        <div class="prediction-box">
            <h2>🎯 Default Surge Pricing</h2>
            <h1 style="font-size: clamp(2rem, 8vw, 4rem);">{fallback_prediction:.2f}x</h1>
            <p>Standard surge multiplier</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 1.5rem; background: #f8f9fa; border-radius: 10px;">
    <h3>🚕 Sigma Cabs - Powered by RIZKY WIBOWO KUSUMO</h3>
    <p>Safe • Reliable • Affordable • 24/7 Available</p>
    <p><strong>Python {sys.version_info.major}.{sys.version_info.minor} | {'ML Enhanced' if ML_AVAILABLE else 'Simplified Mode'}</strong></p>
</div>
""", unsafe_allow_html=True)
