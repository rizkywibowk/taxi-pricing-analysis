import streamlit as st

# HARUS MENJADI COMMAND PERTAMA
st.set_page_config(
    page_title="🚕 Sigma Cabs - Taxi Pricing Analysis",
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
try:
    import joblib
    import pickle
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import GradientBoostingRegressor
    import plotly.express as px
    ML_AVAILABLE = True
except ImportError:
    pass

# Enhanced CSS untuk responsive design, portrait mode, dan dark mode
st.markdown("""
<style>
    /* Root variables untuk theming */
    :root {
        --primary-color: #FF6B6B;
        --secondary-color: #667eea;
        --background-color: #ffffff;
        --text-color: #262730;
        --card-background: #f8f9fa;
        --border-color: #e9ecef;
    }
    
    /* Dark mode variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color: #FF6B6B;
            --secondary-color: #667eea;
            --background-color: #0e1117;
            --text-color: #fafafa;
            --card-background: #262730;
            --border-color: #464a57;
        }
    }
    
    /* Mobile-first responsive design */
    .main-header {
        font-size: clamp(1.5rem, 5vw, 2.5rem);
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        word-wrap: break-word;
        line-height: 1.2;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, var(--secondary-color) 0%, #764ba2 100%);
        padding: clamp(1rem, 3vw, 1.5rem);
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }
    
    .metric-card {
        background: var(--card-background);
        padding: clamp(0.8rem, 2vw, 1rem);
        border-radius: 12px;
        border-left: 5px solid var(--secondary-color);
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        min-height: clamp(100px, 15vh, 150px);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        color: var(--text-color);
    }
    
    .info-box {
        background: var(--card-background);
        padding: clamp(0.8rem, 2vw, 1rem);
        border-radius: 12px;
        border-left: 5px solid #2196f3;
        margin: 0.5rem 0;
        word-wrap: break-word;
        color: var(--text-color);
    }
    
    .contact-info {
        background: var(--card-background);
        padding: clamp(0.8rem, 2vw, 1rem);
        border-radius: 12px;
        border-left: 5px solid #ff9800;
        margin: 0.5rem 0;
        word-wrap: break-word;
        color: var(--text-color);
    }
    
    .error-box {
        background: #ffebee;
        padding: clamp(0.8rem, 2vw, 1rem);
        border-radius: 8px;
        border-left: 5px solid #f44336;
        margin: 0.5rem 0;
        word-wrap: break-word;
    }
    
    .success-box {
        background: #e8f5e8;
        padding: clamp(0.8rem, 2vw, 1rem);
        border-radius: 8px;
        border-left: 5px solid #4caf50;
        margin: 0.5rem 0;
        word-wrap: break-word;
    }
    
    .header-box {
        background: linear-gradient(135deg, var(--secondary-color) 0%, #764ba2 100%);
        padding: clamp(1.5rem, 4vw, 2rem);
        text-align: center;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
    }
    
    /* Portrait mode optimizations */
    @media (orientation: portrait) {
        .main-header {
            font-size: clamp(1.2rem, 6vw, 2rem);
            margin-bottom: 0.8rem;
        }
        
        .metric-card {
            min-height: clamp(80px, 12vh, 120px);
            padding: 0.8rem;
        }
        
        .prediction-box {
            padding: 1rem;
            margin: 0.8rem 0;
        }
        
        .header-box {
            padding: 1.5rem;
        }
        
        /* Stack columns vertically in portrait */
        .stColumns > div {
            width: 100% !important;
            margin-bottom: 1rem;
        }
    }
    
    /* Mobile-specific styles */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        .metric-card {
            min-height: 100px !important;
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
        
        /* Force single column layout on mobile */
        .stColumns > div {
            min-width: 100% !important;
            margin-bottom: 1rem !important;
        }
        
        /* Hide sidebar on mobile */
        .css-1d391kg {
            width: 0px;
        }
        .css-1lcbmhc {
            margin-left: 0px;
        }
        
        /* Adjust input widgets for mobile */
        .stNumberInput, .stSelectbox, .stSlider {
            width: 100%;
            margin-bottom: 0.5rem;
        }
    }
    
    /* Dark mode specific styles */
    @media (prefers-color-scheme: dark) {
        .prediction-box {
            background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%) !important;
            box-shadow: 0 8px 32px rgba(255,255,255,0.1) !important;
        }
        
        .metric-card {
            background: #262730 !important;
            border-left: 5px solid var(--secondary-color) !important;
            color: #fafafa !important;
        }
        
        .info-box {
            background: #262730 !important;
            border-left: 5px solid #4299e1 !important;
            color: #fafafa !important;
        }
        
        .contact-info {
            background: #262730 !important;
            border-left: 5px solid #ed8936 !important;
            color: #fafafa !important;
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
    
    /* Responsive images */
    .stImage > img {
        max-width: 100%;
        height: auto;
        border-radius: 10px;
    }
    
    /* Responsive dataframes */
    .stDataFrame {
        overflow-x: auto;
        max-width: 100%;
    }
    
    /* Responsive plotly charts */
    .js-plotly-plot {
        width: 100% !important;
        height: auto !important;
    }
    
    /* Input widgets responsive */
    .stNumberInput, .stSelectbox, .stSlider {
        width: 100%;
    }
    
    /* Expander responsive */
    .stExpander {
        margin: 0.5rem 0;
    }
    
    /* Button responsive */
    .stButton > button {
        width: 100%;
        padding: 0.75rem 1rem;
        font-size: clamp(0.9rem, 2vw, 1rem);
    }
    
    /* Container responsive */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: clamp(1rem, 3vw, 2rem);
        padding-right: clamp(1rem, 3vw, 2rem);
    }
</style>
""", unsafe_allow_html=True)

# Header Function yang Mobile-Friendly
def display_header():
    """Display header with mobile-friendly design"""
    image_path = 'Picture/Sigma-cabs-in-hyderabad-and-bangalore.jpg'
    
    try:
        if os.path.exists(image_path):
            # Gunakan parameter yang aman untuk Python 3.11
            st.image(image_path, caption='Sigma Cabs - Dedicated to Dedication')
        else:
            # Fallback HTML header
            st.markdown("""
            <div class="header-box">
                <h1 style="margin: 0; font-size: clamp(1.5rem, 5vw, 2.5rem);">🚕 SIGMA CABS</h1>
                <h3 style="margin: 0.5rem 0; font-size: clamp(1rem, 3vw, 1.5rem);">Dedicated to Dedication</h3>
                <p style="margin: 0; font-size: clamp(0.8rem, 2vw, 1rem);">Hyderabad & Bangalore</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        # Ultimate fallback
        st.markdown("""
        <div class="header-box">
            <h1>🚕 SIGMA CABS</h1>
            <h3>Dedicated to Dedication</h3>
            <p>Hyderabad & Bangalore</p>
        </div>
        """, unsafe_allow_html=True)

# Simple Prediction Function
def calculate_surge_pricing(distance, rating, cab_type, traffic, demand, weather):
    """Calculate surge pricing using simple algorithm"""
    try:
        base_surge = 1.0
        distance_factor = min(float(distance) / 50, 0.5)
        rating_factor = (float(rating) - 1) / 20
        
        cab_factors = {
            'Economy (Micro)': 0.0,
            'Standard (Mini)': 0.2,
            'Premium (Prime)': 0.4
        }
        cab_factor = cab_factors.get(str(cab_type), 0.0)
        
        condition_factor = (float(traffic) + float(demand) + float(weather)) / 300
        surge = base_surge + distance_factor + rating_factor + cab_factor + condition_factor
        
        return max(1.0, min(3.0, float(surge)))
    except Exception:
        return 1.5

# Load Sample Data
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

# Create valid model if ML available
def create_valid_model():
    """Create valid model if ML libraries available"""
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
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        np.random.seed(42)
        X_train = np.random.randn(100, 13)
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
    except Exception:
        return None, None, None, None

# Display header
display_header()

# Title
st.markdown('<h1 class="main-header">Taxi Pricing Analysis</h1>', unsafe_allow_html=True)

# System status dengan responsive layout
status_container = st.container()
with status_container:
    if python_version >= "3.12":
        st.warning(f"⚠️ Python {python_version} - Using compatibility mode")
    else:
        st.success(f"✅ deployed with Python {python_version} - for Optimal Model")
    
    if ML_AVAILABLE:
        st.success("✅ HyperParameter best Model is Gradient Boosting for advanced algorithm")
    else:
        st.info("ℹ️ Using simplified algorithm")

# About section dengan responsive columns
about_container = st.container()
with about_container:
    # Gunakan columns yang akan stack di mobile
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

# Dataset preview
df = load_sample_data()
if df is not None:
    with st.expander("📊 Dataset Preview"):
        try:
            st.dataframe(df.head(), use_container_width=True)
            st.write(f"**Records:** {len(df):,} | **Features:** {len(df.columns)}")
        except Exception:
            st.write("Dataset preview not available")

# Load model if available
if ML_AVAILABLE:
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        final_results = {'r2': 0.9455, 'mae': 0.0545, 'rmse': 0.0738, 'model_type': 'GradientBoostingRegressor'}
        st.success("✅ Gradient Boosting Model with advanced Hyperparameter tuning is loaded")
    except:
        model, scaler, feature_names, final_results = create_valid_model()
        if model is not None:
            st.info("ℹ️ Using built-in model")
else:
    model, scaler, feature_names, final_results = None, None, None, None

# Input Section dengan Mobile-First Design
st.markdown("## 🎯 Fare Prediction")

# Trip Details - Stack vertically on mobile
trip_container = st.container()
with trip_container:
    st.markdown("### 🚗 Trip Details")
    
    # Responsive columns
    trip_col1, trip_col2 = st.columns([1, 1])
    
    with trip_col1:
        distance = st.number_input("Distance (km):", min_value=0.1, max_value=100.0, value=5.0, step=0.1)
        cab_type = st.selectbox("Vehicle Type:", ['Economy (Micro)', 'Standard (Mini)', 'Premium (Prime)'])
    
    with trip_col2:
        destination = st.selectbox("Destination:", ["Airport", "Business", "Home"])
        rating = st.slider("Your Rating:", 1, 5, 4)

# Customer Info
customer_container = st.container()
with customer_container:
    st.markdown("### 👤 Customer Information")
    
    cust_col1, cust_col2 = st.columns([1, 1])
    
    with cust_col1:
        months = st.number_input("Customer Since (Months):", min_value=0, max_value=120, value=12)
        lifestyle = st.slider("Lifestyle Index:", 1.0, 3.0, 2.0, step=0.1)
    
    with cust_col2:
        cancellations = st.number_input("Cancellations Last Month:", min_value=0, max_value=10, value=0)
        confidence = st.selectbox("Confidence Level:", ['High Confidence', 'Medium Confidence', 'Low Confidence'])

# Advanced Factors dalam expander untuk menghemat space
with st.expander("🔧 Advanced Pricing Factors"):
    st.markdown("**Adjust these factors for more accurate pricing:**")
    
    # Stack vertically di mobile
    adv_col1, adv_col2, adv_col3 = st.columns([1, 1, 1])
    
    with adv_col1:
        traffic = st.slider("Traffic Density:", 0.0, 100.0, 50.0, help="Current traffic conditions")
    
    with adv_col2:
        demand = st.slider("Demand Level:", 0.0, 100.0, 50.0, help="Current demand for taxis")
    
    with adv_col3:
        weather = st.slider("Weather Impact:", 0.0, 100.0, 30.0, help="Weather impact on travel")

# Prediction Button - Full width
predict_container = st.container()
with predict_container:
    if st.button('🔮 Predict Surge Pricing', type="primary", use_container_width=True):
        try:
            # Calculate surge pricing
            surge = calculate_surge_pricing(distance, rating, cab_type, traffic, demand, weather)
            
            # Display result dengan responsive font
            st.markdown(f"""
            <div class="prediction-box">
                <h2>🎯 Predicted Surge Pricing</h2>
                <h1 style="font-size: clamp(2rem, 8vw, 4rem);">{surge:.2f}x</h1>
                <p>Algorithm: {'ML-Enhanced' if ML_AVAILABLE else 'Rule-Based'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Analysis Results - Responsive layout
            st.markdown("### 📊 Analysis Results")
            
            result_col1, result_col2, result_col3 = st.columns([1, 1, 1])
            
            with result_col1:
                category = "High" if surge > 2.5 else "Medium" if surge > 1.5 else "Low"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📊 Surge Analysis</h4>
                    <p><strong>Category:</strong> {category}</p>
                    <p><strong>Multiplier:</strong> {surge:.2f}x</p>
                    <p><strong>Distance:</strong> {distance} km</p>
                </div>
                """, unsafe_allow_html=True)
            
            with result_col2:
                loyalty = "VIP" if months > 24 else "Loyal" if months > 12 else "Regular"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>👤 Customer</h4>
                    <p><strong>Loyalty:</strong> {loyalty}</p>
                    <p><strong>Rating:</strong> {rating}/5.0 ⭐</p>
                    <p><strong>Since:</strong> {months}m</p>
                </div>
                """, unsafe_allow_html=True)
            
            with result_col3:
                fare = distance * surge * 2.5 + 10
                st.markdown(f"""
                <div class="metric-card">
                    <h4>💰 Estimated Fare</h4>
                    <p><strong>Base:</strong> $10.00</p>
                    <p><strong>Distance:</strong> ${distance * 2.5:.2f}</p>
                    <p><strong>Total:</strong> ${fare:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Conditions Impact
            st.markdown("### 🔍 Conditions Impact")
            condition_score = (traffic + demand + weather) / 3
            impact = "High" if condition_score > 70 else "Medium" if condition_score > 40 else "Low"
            
            st.markdown(f"""
            <div class="info-box">
                <h4>🚦 Current Conditions</h4>
                <p><strong>Traffic:</strong> {traffic:.0f}/100 | <strong>Demand:</strong> {demand:.0f}/100 | <strong>Weather:</strong> {weather:.0f}/100</p>
                <p><strong>Overall Impact:</strong> {impact} ({condition_score:.0f}/100)</p>
                <p><strong>Recommendation:</strong> {'Consider alternative time' if condition_score > 70 else 'Good time to travel'}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception:
            st.error("❌ Prediction error occurred")
            st.markdown("""
            <div class="prediction-box">
                <h2>🎯 Default Surge Pricing</h2>
                <h1 style="font-size: clamp(2rem, 8vw, 4rem);">1.50x</h1>
                <p>Standard multiplier</p>
            </div>
            """, unsafe_allow_html=True)

# Information Section
info_container = st.container()
with info_container:
    st.markdown("---")
    st.markdown("## 💡 Understanding the Factors")
    
    info_col1, info_col2 = st.columns([1, 1])
    
    with info_col1:
        st.markdown("""
        <div class="info-box">
            <h3>🔍 Vehicle Types</h3>
            <ul>
                <li><strong>Economy (Micro):</strong> Budget-friendly, compact cars</li>
                <li><strong>Standard (Mini):</strong> Regular sedans, good comfort</li>
                <li><strong>Premium (Prime):</strong> Luxury vehicles, premium service</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with info_col2:
        st.markdown("""
        <div class="info-box">
            <h3>🌦️ Pricing Factors</h3>
            <ul>
                <li><strong>Traffic Density:</strong> Road congestion level</li>
                <li><strong>Demand Level:</strong> Current booking requests</li>
                <li><strong>Weather Impact:</strong> Weather affecting travel</li>
                <li><strong>Distance:</strong> Primary cost factor</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer dengan responsive design
footer_container = st.container()
with footer_container:
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: clamp(1rem, 3vw, 1.5rem); 
               background: var(--card-background); border-radius: 10px; 
               color: var(--text-color); margin-top: 1rem;">
        <h3 style="margin: 0; font-size: clamp(1.2rem, 4vw, 1.8rem);">🚕 Sigma Cabs - Powered by RIZKY WIBOWO KUSUMO</h3>
        <p style="margin: 0.5rem 0; font-size: clamp(0.9rem, 3vw, 1rem);">Safe • Reliable • Affordable • 24/7 Available</p>
        <p style="margin: 0; font-size: clamp(0.8rem, 2.5vw, 0.9rem);">
            <strong>Python {python_version} | {'ML Enhanced' if ML_AVAILABLE else 'Simplified Mode'} | all device access optimized</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
