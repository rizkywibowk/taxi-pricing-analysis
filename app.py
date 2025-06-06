import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
import warnings
import plotly.express as px

# HARUS MENJADI COMMAND PERTAMA
st.set_page_config(
    page_title="🚕 Sigma Cabs - Taxi Pricing Analysis",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import setelah set_page_config
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC  # Import Support Vector Machine (SVM)

warnings.filterwarnings('ignore')

# Check Python version
python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

# Try import ML libraries dengan error handling
ML_AVAILABLE = False
try:
    ML_AVAILABLE = True
except ImportError:
    pass

# Enhanced CSS untuk theming dan penataan halaman
st.markdown("""
<style>
    /* Custom CSS styling for presentation */
    :root {
        --primary-color: #FF6B6B;
        --secondary-color: #667eea;
        --background-color: #ffffff;
        --text-color: #262730;
        --card-background: #f8f9fa;
        --border-color: #e9ecef;
    }
    /* Dark mode support */
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
    .main-header {
        font-size: clamp(2rem, 6vw, 3rem);
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Display header with an image
def display_header():
    image_path = 'Picture/Sigma-cabs-in-hyderabad-and-bangalore.jpg'  # Set the image path here
    
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

# Function to preprocess input data
def preprocess_data(data):
    # Encode 'cab_type' into numerical values
    data['cab_type'] = data['cab_type'].map({'Economy (Micro)': 0, 'Standard (Mini)': 1, 'Premium (Prime)': 2})
    
    # Feature scaling (if necessary, for example if using SVM with kernels)
    scaler = joblib.load('Model for Streamlit/scaler.pkl')  # Load your scaler
    data = scaler.transform(data)
    
    return data

# Load the SVM model
svm_model = joblib.load('Model for Streamlit/svm_model.pkl')

# Load the dataset (from Dataset folder)
def load_dataset():
    try:
        df = pd.read_csv('Dataset/sigma_cabs.csv')
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Display Header and Introduction
display_header()
st.markdown("<h1 class='main-header'>🎯 Intelligent Taxi Pricing Analysis</h1>", unsafe_allow_html=True)

# Dataset preview
df = load_dataset()
if df is not None:
    with st.expander("📊 Dataset Preview"):
        try:
            st.dataframe(df.head(), use_container_width=True)
            st.write(f"**Records:** {len(df):,} | **Features:** {len(df.columns)}")
        except Exception:
            st.write("Dataset preview not available")

# Get user input for trip prediction
st.markdown("## 🎯 Intelligent Fare Prediction")

# Trip Details Input
distance = st.number_input("🛣️ Distance (km):", min_value=0.1, max_value=100.0, value=5.0, step=0.1)
rating = st.slider("⭐ Customer Rating:", 1, 5, 4)
cab_type = st.selectbox("🚙 Vehicle Type:", ['Economy (Micro)', 'Standard (Mini)', 'Premium (Prime)'])
traffic = st.slider("🚦 Traffic Density:", 0.0, 100.0, 50.0)
demand = st.slider("📈 Demand Level:", 0.0, 100.0, 50.0)
weather = st.slider("🌧️ Weather Impact:", 0.0, 100.0, 30.0)

# Collect the input data into a dictionary
user_input = {
    'distance': distance,
    'rating': rating,
    'cab_type': cab_type,
    'traffic': traffic,
    'demand': demand,
    'weather': weather
}

# Predict with SVM model
if st.button("Predict Surge Pricing"):
    input_data = pd.DataFrame([user_input])
    
    # Preprocess input data
    input_data_preprocessed = preprocess_data(input_data)
    
    # Make prediction
    surge_price = svm_model.predict(input_data_preprocessed)
    st.write(f"Predicted Surge Pricing Multiplier: {surge_price[0]:.2f}x")

# Calculate surge pricing based on real-time conditions
def calculate_surge_pricing(distance, rating, cab_type, traffic, demand, weather):
    base_surge = 1.0
    distance_factor = min(distance / 50, 0.5)
    rating_factor = (rating - 1) / 20
    cab_factor = {'Economy (Micro)': 0.0, 'Standard (Mini)': 0.2, 'Premium (Prime)': 0.4}.get(cab_type, 0.0)
    condition_factor = (traffic + demand + weather) / 300
    surge = base_surge + distance_factor + rating_factor + cab_factor + condition_factor
    return max(1.0, min(3.0, surge))

# Display surge pricing breakdown
if st.button('🔮 Calculate Smart Pricing', type="primary", use_container_width=True):
    surge_breakdown = calculate_surge_pricing(distance, rating, cab_type, traffic, demand, weather)
    st.markdown(f"""
    <div class="prediction-box">
        <h2>🎯 Predicted Surge Pricing</h2>
        <h1>{surge_breakdown:.2f}x</h1>
        <p>The increased fare multiplier due to current conditions</p>
    </div>
    """, unsafe_allow_html=True)
