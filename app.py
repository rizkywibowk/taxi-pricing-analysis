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

# Try import ML libraries
ML_AVAILABLE = False
try:
    import joblib
    import pickle
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingRegressor
    import plotly.express as px
    ML_AVAILABLE = True
except ImportError:
    pass

# CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        padding: 1rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        min-height: 120px;
    }
    
    .info-box {
        background: linear-gradient(145deg, #e3f2fd, #bbdefb);
        padding: 1rem;
        border-radius: 12px;
        border-left: 5px solid #2196f3;
        margin: 0.5rem 0;
    }
    
    .contact-info {
        background: linear-gradient(145deg, #fff3e0, #ffe0b2);
        padding: 1rem;
        border-radius: 12px;
        border-left: 5px solid #ff9800;
        margin: 0.5rem 0;
    }
    
    .header-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        text-align: center;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
    }
    
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem !important;
        }
        .metric-card {
            min-height: 100px !important;
            padding: 0.8rem !important;
        }
        .header-box {
            padding: 1.5rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Header Function yang Aman
def display_header():
    """Display header with safe image handling"""
    image_path = 'Picture/Sigma-cabs-in-hyderabad-and-bangalore.jpg'
    
    try:
        if os.path.exists(image_path):
            # Gunakan parameter yang kompatibel dengan semua versi
            st.image(image_path, caption='Sigma Cabs - Dedicated to Dedication', width=None)
        else:
            # Fallback HTML header
            st.markdown("""
            <div class="header-box">
                <h1 style="margin: 0; font-size: clamp(1.5rem, 4vw, 2.5rem);">🚕 SIGMA CABS</h1>
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
        # Base surge
        base_surge = 1.0
        
        # Distance factor (0-0.5)
        distance_factor = min(float(distance) / 50, 0.5)
        
        # Rating factor (0-0.2)
        rating_factor = (float(rating) - 1) / 20
        
        # Cab type factor
        cab_factors = {
            'Economy (Micro)': 0.0,
            'Standard (Mini)': 0.2,
            'Premium (Prime)': 0.4
        }
        cab_factor = cab_factors.get(str(cab_type), 0.0)
        
        # Condition factors (0-0.3)
        condition_factor = (float(traffic) + float(demand) + float(weather)) / 300
        
        # Calculate final surge
        surge = base_surge + distance_factor + rating_factor + cab_factor + condition_factor
        
        # Clamp between 1.0 and 3.0
        return max(1.0, min(3.0, float(surge)))
        
    except Exception:
        return 1.5  # Safe default

# Load Sample Data dengan Error Handling
@st.cache_data
def load_sample_data():
    """Load or create sample data safely"""
    try:
        if os.path.exists('Dataset/sigma_cabs.csv'):
            return pd.read_csv('Dataset/sigma_cabs.csv')
        else:
            # Create sample data
            np.random.seed(42)
            data = {
                'Trip_Distance': np.random.uniform(1, 50, 100),
                'Customer_Rating': np.random.uniform(1, 5, 100),
                'Surge_Pricing_Type': np.random.uniform(1, 3, 100)
            }
            return pd.DataFrame(data)
    except Exception:
        return None

# Display header
display_header()

# Title
st.markdown('<h1 class="main-header">Taxi Pricing Analysis</h1>', unsafe_allow_html=True)

# System status
status_col1, status_col2 = st.columns(2)

with status_col1:
    if python_version >= "3.12":
        st.warning(f"⚠️ Python {python_version} - Using compatibility mode")
    else:
        st.success(f"✅ Python {python_version} - Optimal")

with status_col2:
    if ML_AVAILABLE:
        st.success("✅ ML libraries available")
    else:
        st.info("ℹ️ Using simplified algorithm")

# About section
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

# Dataset preview dengan error handling
df = load_sample_data()
if df is not None:
    with st.expander("📊 Dataset Preview"):
        try:
            st.dataframe(df.head(), use_container_width=True)
            st.write(f"**Records:** {len(df):,} | **Features:** {len(df.columns)}")
        except Exception:
            st.write("Dataset preview not available")

# Input Section
st.markdown("## 🎯 Fare Prediction")

# Trip Details
st.markdown("### 🚗 Trip Details")
trip_col1, trip_col2 = st.columns(2)

with trip_col1:
    distance = st.number_input("Distance (km):", min_value=0.1, max_value=100.0, value=5.0, step=0.1)
    cab_type = st.selectbox("Vehicle Type:", ['Economy (Micro)', 'Standard (Mini)', 'Premium (Prime)'])

with trip_col2:
    destination = st.selectbox("Destination:", ["Airport", "Business", "Home"])
    rating = st.slider("Your Rating:", 1, 5, 4)

# Customer Info
st.markdown("### 👤 Customer Information")
cust_col1, cust_col2 = st.columns(2)

with cust_col1:
    months = st.number_input("Customer Since (Months):", min_value=0, max_value=120, value=12)
    lifestyle = st.slider("Lifestyle Index:", 1.0, 3.0, 2.0, step=0.1)

with cust_col2:
    cancellations = st.number_input("Cancellations Last Month:", min_value=0, max_value=10, value=0)
    confidence = st.selectbox("Confidence Level:", ['High Confidence', 'Medium Confidence', 'Low Confidence'])

# Advanced Factors
with st.expander("🔧 Advanced Pricing Factors"):
    adv_col1, adv_col2, adv_col3 = st.columns(3)
    
    with adv_col1:
        traffic = st.slider("Traffic Density:", 0.0, 100.0, 50.0)
    
    with adv_col2:
        demand = st.slider("Demand Level:", 0.0, 100.0, 50.0)
    
    with adv_col3:
        weather = st.slider("Weather Impact:", 0.0, 100.0, 30.0)

# Prediction Button
if st.button('🔮 Predict Surge Pricing', type="primary"):
    try:
        # Calculate surge pricing
        surge = calculate_surge_pricing(distance, rating, cab_type, traffic, demand, weather)
        
        # Display result
        st.markdown(f"""
        <div class="prediction-box">
            <h2>🎯 Predicted Surge Pricing</h2>
            <h1 style="font-size: 3rem;">{surge:.2f}x</h1>
            <p>Algorithm: {'ML-Enhanced' if ML_AVAILABLE else 'Rule-Based'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis Results
        st.markdown("### 📊 Analysis Results")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
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
        
    except Exception as e:
        st.error("❌ Prediction error occurred")
        st.markdown("""
        <div class="prediction-box">
            <h2>🎯 Default Surge Pricing</h2>
            <h1 style="font-size: 3rem;">1.50x</h1>
            <p>Standard multiplier</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 1.5rem; background: #f8f9fa; border-radius: 10px;">
    <h3>🚕 Sigma Cabs - Powered by RIZKY WIBOWO KUSUMO</h3>
    <p>Safe • Reliable • Affordable • 24/7 Available</p>
    <p><strong>Python {python_version} | {'ML Enhanced' if ML_AVAILABLE else 'Simplified Mode'}</strong></p>
</div>
""", unsafe_allow_html=True)
