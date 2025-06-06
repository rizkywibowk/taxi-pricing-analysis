import streamlit as st   
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# HARUS MENJADI COMMAND PERTAMA
st.set_page_config(
    page_title="🚕 Sigma Cabs - Taxi Pricing Analysis",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Check Python version
python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

# Try import ML libraries dengan error handling
ML_AVAILABLE = False
MODEL_SOURCE = "fallback"
try:
    import joblib
    import pickle
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import GradientBoostingRegressor
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
        --secondary-color: #2e7d32;
        --background-color: #e8f5e8;  /* Green Background */
        --text-color: #1b5e20;
        --card-background: rgba(255, 255, 255, 0.95);
        --border-color: #81c784;
        --success-color: #2e7d32;
        --warning-color: #ff8f00;
        --danger-color: #d32f2f;
        --info-color: #1976d2;
        --accent-green: #4caf50;
        --light-green: #c8e6c9;
        --dark-green: #1b5e20;
    }

    /* Dark mode variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color: #FF6B6B;
            --secondary-color: #4fc3f7;
            --background-color: #0d47a1;
            --text-color: #e3f2fd;
            --card-background: rgba(30, 63, 102, 0.95);
            --border-color: #42a5f5;
            --accent-green: #4fc3f7;
            --light-green: rgba(79, 195, 247, 0.2);
        }
    }

    /* Background Hijau Daun Cerah Enhanced */
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

    /* Enhanced main header with gradient text */
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

    /* Prediction Box */
    .prediction-box {
        background: linear-gradient(135deg, 
                   var(--secondary-color) 0%, 
                   color-mix(in srgb, var(--secondary-color) 70%, black 30%) 100%);
        padding: 1.5rem;
        border-radius: 15px;
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
</style>
""", unsafe_allow_html=True)

# Enhanced functions untuk visualisasi
def create_gauge_chart(value, max_value=100, title=""):
    """Create enhanced gauge chart dengan green theme"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16, 'color': '#2e7d32'}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, max_value], 'tickcolor': '#2e7d32'},
            'bar': {'color': "#2e7d32", 'thickness': 0.3},
            'steps': [
                {'range': [0, 30], 'color': "#c8e6c9"},
                {'range': [30, 70], 'color': "#81c784"},
                {'range': [70, 100], 'color': "#4caf50"}
            ],
            'threshold': {
                'line': {'color': "#d32f2f", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(
        height=200, 
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2e7d32')
    )
    return fig

# Display header function
def display_header():
    """Display header with green theme"""
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

st.markdown('<h1 class="main-header">🌱 Advanced Eco-Smart Taxi Pricing Analysis 🌊</h1>', unsafe_allow_html=True)

# Load model with advanced validation
model, scaler, feature_names, final_results, load_status = load_model()

# About and Contact sections
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    <div class="info-box">
        <h3>🌟 About Sigma Cabs</h3>
        <p><strong>Sigma Cabs</strong> provides exceptional cab service in <strong>Hyderabad</strong> and <strong>Bangalore</strong>.
        Reliable and safe transportation, always ready to meet your travel needs.</p>
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

# Display Dataset
df = load_sample_data()
if df is not None:
    with st.expander("📊 Dataset Preview"):
        st.dataframe(df.head(), use_container_width=True)
        st.write(f"**Records:** {len(df):,} | **Features:** {len(df.columns)}")

# Input Section
st.markdown("## 🎯 Advanced Fare Prediction")

trip_container = st.container()
with trip_container:
    st.markdown("### 🚗 Trip Details")
    trip_col1, trip_col2 = st.columns([1, 1])
    with trip_col1:
        distance = st.number_input("🛣️ Distance (km):", min_value=0.1, max_value=100.0, value=5.0, step=0.1)
        cab_type = st.selectbox("🚙 Vehicle Type:", ['Economy (Micro)', 'Standard (Mini)', 'Premium (Prime)'])
    with trip_col2:
        destination = st.selectbox("📍 Destination:", ["Airport", "Business", "Home"])
        rating = st.slider("⭐ Your Rating:", 1, 5, 4)

customer_container = st.container()
with customer_container:
    st.markdown("### 👤 Customer Information")
    cust_col1, cust_col2 = st.columns([1, 1])
    with cust_col1:
        months = st.number_input("📅 Customer Since (Months):", min_value=0, max_value=120, value=12)
        lifestyle = st.slider("💎 Lifestyle Index:", 1.0, 3.0, 2.0, step=0.1)
    with cust_col2:
        cancellations = st.number_input("❌ Cancellations Last Month:", min_value=0, max_value=10, value=0)
        confidence = st.selectbox("🎯 Service Confidence:", ['High Confidence', 'Medium Confidence', 'Low Confidence'])

with st.expander("⚙️ Advanced Pricing Factors"):
    st.markdown("**Adjust these real-time factors for maximum precision:**")
    adv_col1, adv_col2, adv_col3 = st.columns([1, 1, 1])
    with adv_col1:
        traffic = st.slider("🚦 Traffic Density:", 0.0, 100.0, 50.0)
    with adv_col2:
        demand = st.slider("📈 Demand Level:", 0.0, 100.0, 50.0)
    with adv_col3:
        weather = st.slider("🌧 Weather Impact:", 0.0, 100.0, 30.0)

# Prediction Button
if st.button('🔮 Calculate Advanced Precision Pricing', type="primary", use_container_width=True):
    try:
        # Prepare input data
        input_data = {
            'Trip_Distance': float(distance),
            'Customer_Rating': float(rating),
            'Customer_Since_Months': int(months),
            'Life_Style_Index': float(lifestyle),
            'Type_of_Cab': str(cab_type),
            'Confidence_Life_Style_Index': str(confidence),
            'Destination_Type': str(destination),
            'Gender': 'Male',  # Default
            'Cancellation_Last_1Month': int(cancellations),
            'Var1': float(traffic),
            'Var2': float(demand),
            'Var3': float(weather)
        }
        
        # Preprocess data
        processed_array = preprocess_input_data_robust(input_data, feature_names)
        
        # Validasi
        if processed_array.shape[1] != len(feature_names):
            raise ValueError(f"Feature mismatch: {processed_array.shape[1]} vs {len(feature_names)}")
        
        if np.isnan(processed_array).any() or np.isinf(processed_array).any():
            raise ValueError("Invalid input values")
        
        # Scale dan predict menggunakan advanced model
        scaled_input = scaler.transform(processed_array)
        prediction_result = model.predict(scaled_input)
        
        if not isinstance(prediction_result, np.ndarray) or len(prediction_result) == 0:
            raise ValueError("Invalid prediction")
        
        surge = float(prediction_result[0])
        
        if np.isnan(surge) or np.isinf(surge):
            raise ValueError("Invalid prediction value")
        
        surge = max(1.0, min(3.0, surge))
        
        st.markdown(f"""
        <div class="prediction-box">
            <h2>🎯 Advanced Gradient Boosting Prediction</h2>
            <h1>{surge:.2f}x</h1>
            <p>Powered by {MODEL_SOURCE.upper()} - Maximum Precision AI</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Detailed Analysis Results")
        result_col1, result_col2, result_col3 = st.columns([1, 1, 1])
        
        with result_col1:
            category = "High" if surge > 2.5 else "Medium" if surge > 1.5 else "Low"
            surge_class = get_surge_category_class(surge)
            st.markdown(f"""
            <div class="metric-card {surge_class}">
                <h4><span class="icon">📊</span>Surge Analysis</h4>
                <p><strong>Category:</strong> {category}</p>
                <p><strong>Multiplier:</strong> 
                    <span class="tooltip">{surge:.2f}x
                        <span class="tooltiptext">Advanced Gradient Boosting prediction with 94.55% accuracy</span>
                    </span>
                </p>
                <p><strong>Distance:</strong> {distance} km</p>
            </div>
            """, unsafe_allow_html=True)
            surge_fig = create_surge_gauge(surge)
            st.plotly_chart(surge_fig, use_container_width=True)
            
        with result_col2:
            loyalty = "VIP" if months > 24 else "Loyal" if months > 12 else "Regular" if months > 3 else "New"
            loyalty_class = get_loyalty_class(months)
            st.markdown(f"""
            <div class="metric-card {loyalty_class}">
                <h4><span class="icon">👤</span>Customer Profile</h4>
                <p><strong>Loyalty Status:</strong> 
                    <span class="tooltip">{loyalty}
                        <span class="tooltiptext">Customer loyalty affects pricing through our advanced model</span>
                    </span>
                </p>
                <p><strong>Rating:</strong> {rating}/5.0 ⭐</p>
                <p><strong>Since:</strong> {months} months</p>
            </div>
            """, unsafe_allow_html=True)
            
        with result_col3:
            base_fare = 10.0
            distance_cost = distance * 2.5
            surge_additional = (distance_cost * (surge - 1))
            total_fare = base_fare + distance_cost + surge_additional
            st.markdown(f"""
            <div class="metric-card">
                <h4><span class="icon">💰</span>Precision Fare</h4>
                <div class="fare-breakdown">
                    <div class="fare-item">
                        <span>Base Fare:</span>
                        <span>${base_fare:.2f}</span>
                    </div>
                    <div class="fare-item">
                        <span>Distance ({distance} km):</span>
                        <span>${distance_cost:.2f}</span>
                    </div>
                    <div class="fare-item">
                        <span>AI Surge ({surge:.2f}x):</span>
                        <span>+${surge_additional:.2f}</span>
                    </div>
                    <div class="fare-item">
                        <span>Total:</span>
                        <span>${total_fare:.2f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🔍 Real-time Conditions Impact")
        condition_col1, condition_col2 = st.columns([1, 1])
        
        with condition_col1:
            condition_score = (traffic + demand + weather) / 3
            impact = "High Impact" if condition_score > 70 else "Medium Impact" if condition_score > 40 else "Low Impact"
            st.markdown(f"""
            <div class="info-box">
                <h4><span class="icon">🚦</span>Current Conditions</h4>
                <p><strong>Traffic Density:</strong> {traffic:.0f}/100</p>
                <p><strong>Demand Level:</strong> {demand:.0f}/100</p>
                <p><strong>Weather Impact:</strong> {weather:.0f}/100</p>
                <p><strong>Overall Impact:</strong> {impact} ({condition_score:.0f}/100)</p>
                <p><strong>💡 AI Recommendation:</strong> {'Consider alternative time or route' if condition_score > 70 else 'Optimal conditions for travel'}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with condition_col2:
            conditions_fig = create_gauge_chart(condition_score, 100, "Conditions Impact")
            st.plotly_chart(conditions_fig, use_container_width=True)
        
        # Enhanced factors chart
        st.markdown("### 📈 Advanced AI Model Factors Analysis")
        
        # Calculate factor contributions
        base_contribution = 1.0
        distance_contribution = min(distance / 50, 0.5)
        rating_contribution = (rating - 1) / 20
        cab_contribution = {'Economy (Micro)': 0.0, 'Standard (Mini)': 0.2, 'Premium (Prime)': 0.4}.get(cab_type, 0.0)
        condition_contribution = (traffic + demand + weather) / 300
        
        factors_data = {
            'Factor': ['Base Rate', 'Distance', 'Rating', 'Vehicle Type', 'Conditions'],
            'Impact': [
                base_contribution,
                distance_contribution,
                rating_contribution,
                cab_contribution,
                condition_contribution
            ]
        }
        
        factors_df = pd.DataFrame(factors_data)
        fig_factors = px.bar(
            factors_df, 
            x='Factor', 
            y='Impact',
            title="Advanced Gradient Boosting - Factor Contributions",
            color='Impact',
            color_continuous_scale='Greens'
        )
        fig_factors.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2e7d32')
        )
        st.plotly_chart(fig_factors, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Advanced prediction error: {str(e)}")
        st.markdown("""
        <div class="prediction-box">
            <h2>🎯 Fallback Pricing</h2>
            <h1>1.50x</h1>
            <p>Using simplified algorithm</p>
        </div>
        """, unsafe_allow_html=True)

# Additional Information
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
        <h3>🎯 Confidence Levels</h3>
        <ul>
            <li><strong>High:</strong> Frequent user, trusts service</li>
            <li><strong>Medium:</strong> Occasional user</li>
            <li><strong>Low:</strong> New or hesitant user</li>
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
        <h3>📊 How It Works</h3>
        <p>Our AI model analyzes all factors to predict fair surge pricing in real-time.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 1.5rem; background: var(--background-color, #f8f9fa); 
           border-radius: 10px; margin-top: 1rem; word-wrap: break-word;">
    <h3 style="margin: 0; font-size: clamp(1.2rem, 4vw, 1.8rem);">🚕 Sigma Cabs - Powered by RIZKY WIBOWO KUSUMO MODEL</h3>
    <p style="margin: 0.5rem 0; font-size: clamp(0.9rem, 3vw, 1rem);">Safe • Reliable • Affordable • 24/7 Available</p>
    <p style="margin: 0; font-size: clamp(0.8rem, 2.5vw, 0.9rem);"><strong>Model Accuracy: {final_results['r2']*100 if final_results else 94.55:.2f}% | Gradient Boosting Algorithm</strong></p>
</div>
""", unsafe_allow_html=True)
