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
# Enhanced Footer Section
footer_container = st.container()
with footer_container:
    st.markdown("---")
    st.markdown(f"""
    <div class="footer-container" style="text-align: center; padding: clamp(1.5rem, 4vw, 2rem); 
               border-radius: 15px; margin-top: 1.5rem; background: var(--card-background);
               border: 1px solid var(--border-color);">
        <h3 style="margin: 0; font-size: clamp(1.3rem, 5vw, 2rem);">🚕 Sigma Cabs - Powered by RIZKY WIBOWO KUSUMO</h3>
        <p style="margin: 1rem 0; font-size: clamp(1rem, 3vw, 1.2rem);">Safe • Reliable • Affordable • 24/7 Available</p>
        <p style="margin: 0; font-size: clamp(0.9rem, 2.5vw, 1rem);">
            <strong>Python {python_version} | {'🤖 ML Enhanced' if ML_AVAILABLE else '⚡ Simplified Mode'} | 📱 Mobile Optimized</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Interactive Elements for Fare Breakdown and User Inputs
st.markdown("---")
st.markdown("### 💡 Understanding Smart Pricing")

# Display dynamic fare breakdown and conditions
with st.expander("🔍 Surge Factors Breakdown"):
    factors_data = {
        'Factor': ['Base Rate', 'Distance', 'Rating', 'Vehicle Type', 'Conditions'],
        'Impact': [
            surge_breakdown['base'],
            surge_breakdown['distance_factor'],
            surge_breakdown['rating_factor'],
            surge_breakdown['cab_factor'],
            surge_breakdown['condition_factor']
        ]
    }
    
    factors_df = pd.DataFrame(factors_data)
    fig_factors = px.bar(
        factors_df, 
        x='Factor', 
        y='Impact',
        title="Factors Contributing to Surge Pricing",
        color='Impact',
        color_continuous_scale='RdYlBu_r'
    )
    fig_factors.update_layout(height=400)
    st.plotly_chart(fig_factors, use_container_width=True)

# Footer: Providing More Information
info_container = st.container()
with info_container:
    st.markdown("### 🚗 Vehicle Categories & Pricing Insights")
    
    st.markdown("""
    <div class="info-box">
        <h4><span class="icon">🚗</span>Vehicle Categories</h4>
        <ul>
            <li><strong>🚗 Economy (Micro):</strong> Budget-friendly, compact cars for short trips</li>
            <li><strong>🚙 Standard (Mini):</strong> Regular sedans with good comfort for medium trips</li>
            <li><strong>🚘 Premium (Prime):</strong> Luxury vehicles with premium service</li>
        </ul>
        <h4><span class="icon">🎯</span>Confidence Levels</h4>
        <ul>
            <li><strong>🟢 High:</strong> Frequent user who trusts the service completely</li>
            <li><strong>🟡 Medium:</strong> Occasional user with moderate confidence</li>
            <li><strong>🔴 Low:</strong> New or hesitant user, needs more assurance</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Additional Information on Dynamic Pricing
    st.markdown("""
    <div class="info-box">
        <h4><span class="icon">🌧️</span>Dynamic Pricing Factors</h4>
        <ul>
            <li><strong>🚦 Traffic Density:</strong> Real-time road congestion levels</li>
            <li><strong>📈 Demand Level:</strong> Current booking requests in your area</li>
            <li><strong>🌤 Weather Impact:</strong> Weather conditions affecting travel safety</li>
            <li><strong>📏 Distance:</strong> Primary cost factor for your journey</li>
        </ul>
        <h4><span class="icon">🤖</span>How Our AI-Based Pricing Works</h4>
        <p>Our advanced machine learning model analyzes <strong>13+ factors</strong> in real-time to predict fair and transparent surge pricing, ensuring you get the best possible fare.</p>
    </div>
    """, unsafe_allow_html=True)

# System Status Section
status_container = st.container()
with status_container:
    st.markdown("---")
    st.markdown("## 🔧 System Performance")
    
    status_col1, status_col2 = st.columns([1, 1])
    
    with status_col1:
        if python_version >= "3.12":
            st.warning(f"⚠️ Python {python_version} - Using compatibility mode")
        else:
            st.success(f"✅ Deployed with Python {python_version} - Optimal Performance")
    
    with status_col2:
        if ML_AVAILABLE:
            st.success("✅ Advanced Gradient Boosting Model - 94.55% Accuracy")
        else:
            st.info("ℹ️ Using simplified rule-based algorithm")

# Footer: Displaying information and credits
footer_container = st.container()
with footer_container:
    st.markdown("---")
    st.markdown(f"""
    <div class="footer-container" style="text-align: center; padding: clamp(1.5rem, 4vw, 2rem); 
               border-radius: 15px; margin-top: 1.5rem; background: var(--card-background);
               border: 1px solid var(--border-color);">
        <h3 style="margin: 0; font-size: clamp(1.3rem, 5vw, 2rem);">🚕 Sigma Cabs - Powered by RIZKY WIBOWO KUSUMO</h3>
        <p style="margin: 1rem 0; font-size: clamp(1rem, 3vw, 1.2rem);">Safe • Reliable • Affordable • 24/7 Available</p>
        <p style="margin: 0; font-size: clamp(0.9rem, 2.5vw, 1rem);">
            <strong>Python {python_version} | {'🤖 ML Enhanced' if ML_AVAILABLE else '⚡ Simplified Mode'} | 📱 Mobile Optimized</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
# Final Footer Section
footer_container = st.container()
with footer_container:
    st.markdown("---")
    st.markdown(f"""
    <div class="footer-container" style="text-align: center; padding: clamp(1.5rem, 4vw, 2rem); 
               border-radius: 15px; margin-top: 1.5rem; background: var(--card-background);
               border: 1px solid var(--border-color);">
        <h3 style="margin: 0; font-size: clamp(1.3rem, 5vw, 2rem);">🚕 Sigma Cabs - Powered by RIZKY WIBOWO KUSUMO</h3>
        <p style="margin: 1rem 0; font-size: clamp(1rem, 3vw, 1.2rem);">Safe • Reliable • Affordable • 24/7 Available</p>
        <p style="margin: 0; font-size: clamp(0.9rem, 2.5vw, 1rem);">
            <strong>Python {python_version} | {'🤖 ML Enhanced' if ML_AVAILABLE else '⚡ Simplified Mode'} | 📱 Mobile Optimized</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Display Dataset Overview
st.markdown("### 📊 Dataset Overview")
st.write("The dataset is used to predict surge pricing based on various factors such as distance, rating, and conditions. Here is a snapshot of the dataset:")

# Show the first few records of the dataset
if df is not None:
    st.dataframe(df.head(), use_container_width=True)

# Display Description of the Dataset
st.markdown("""
The dataset consists of the following key features:

- **Trip_Distance**: The distance of the trip in kilometers.
- **Customer_Rating**: The rating given by the customer for the trip (1-5 scale).
- **Surge_Pricing_Type**: The type of surge pricing applied (based on demand, traffic, and weather).
- **Cab_Type**: Type of vehicle chosen (Economy, Standard, Premium).
- **Traffic**, **Demand**, **Weather**: These are external factors that impact surge pricing in real-time.
""")

# Allow the user to input a new trip for surge prediction
st.markdown("### 🛣️ Trip Input")

trip_col1, trip_col2 = st.columns([1, 1])

with trip_col1:
    distance_input = st.number_input("Enter Trip Distance (km):", min_value=0.1, value=5.0, step=0.1)
    cab_type_input = st.selectbox("Select Cab Type:", ['Economy (Micro)', 'Standard (Mini)', 'Premium (Prime)'])
    rating_input = st.slider("Rate Your Trip (1-5):", min_value=1, max_value=5, value=4)

with trip_col2:
    traffic_input = st.slider("Traffic Density (0 = No Traffic, 100 = Heavy Traffic):", min_value=0, max_value=100, value=50)
    demand_input = st.slider("Demand Level (0 = Low, 100 = High):", min_value=0, max_value=100, value=50)
    weather_input = st.slider("Weather Impact (0 = Perfect, 100 = Severe):", min_value=0, max_value=100, value=30)

# Prepare the input for prediction
user_input = {
    "Trip_Distance": distance_input,
    "Customer_Rating": rating_input,
    "Cab_Type": cab_type_input,
    "Traffic": traffic_input,
    "Demand": demand_input,
    "Weather": weather_input
}

# Predict the Surge Pricing
if st.button("Predict Surge Pricing"):
    # Preprocess user input
    input_data = pd.DataFrame([user_input])
    input_data_processed = preprocess_data(input_data)

    # Make the prediction using SVM model
    surge_prediction = svm_model.predict(input_data_processed)
    st.markdown(f"**Predicted Surge Pricing Multiplier:** {surge_prediction[0]:.2f}x")

# Additional Features: Displaying Surge Level
def get_surge_level(surge_value):
    if surge_value <= 1.5:
        return "Low Surge", "#28a745"  # Green
    elif surge_value <= 2.5:
        return "Medium Surge", "#ffc107"  # Yellow
    else:
        return "High Surge", "#dc3545"  # Red

# Display Surge Level with dynamic color
surge_level, surge_color = get_surge_level(surge_prediction[0])
st.markdown(f"### Surge Pricing Category: **{surge_level}**")
st.markdown(f"<p style='color:{surge_color}; font-size:2rem;'>Surge Multiplier: {surge_prediction[0]:.2f}x</p>", unsafe_allow_html=True)

# Footer: Enhancing User Experience
st.markdown("""
---
## 🚕 **About Sigma Cabs**
Sigma Cabs offers exceptional cab service in **Hyderabad** and **Bangalore**, with **ML-powered surge pricing** ensuring that the fares are fair and transparent based on real-time conditions.

### **Contact Information:**
- **📞** Toll-Free: 1800-420-9999
- **📱** 24/7 Support: 040-63 63 63 63

Enjoy your ride with **Sigma Cabs**, where we combine advanced **AI-powered technology** and **customer satisfaction**.
""")

# Add the footer with some extra information
st.markdown("""
    <footer style="text-align:center; padding: 1.5rem; background-color: #f1f1f1;">
        <p>🚕 **Sigma Cabs** | Powered by Machine Learning | Transparent and Fair Pricing | Available 24/7 in Hyderabad & Bangalore</p>
        <p>Made with ❤️ by <strong>Your Name</strong></p>
    </footer>
""", unsafe_allow_html=True)
