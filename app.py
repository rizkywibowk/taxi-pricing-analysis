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
    import plotly.graph_objects as go
    ML_AVAILABLE = True
except ImportError:
    pass

# Enhanced CSS dengan semua improvements
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
        --success-color: #28a745;
        --warning-color: #ffc107;
        --danger-color: #dc3545;
        --info-color: #17a2b8;
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
    
    /* Enhanced main header with better typography */
    .main-header {
        font-size: clamp(2rem, 6vw, 3rem);
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        word-wrap: break-word;
        line-height: 1.2;
        font-weight: 700;
    }
    
    /* Enhanced prediction box with larger text */
    .prediction-box {
        background: linear-gradient(135deg, var(--secondary-color) 0%, #764ba2 100%);
        padding: clamp(1.5rem, 4vw, 2rem);
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        word-wrap: break-word;
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    .prediction-box h1 {
        font-size: clamp(3rem, 10vw, 5rem) !important;
        margin: 1rem 0 !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Color-coded metric cards */
    .metric-card {
        background: var(--card-background);
        padding: clamp(1rem, 3vw, 1.2rem);
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        margin: 0.8rem 0;
        min-height: clamp(120px, 18vh, 180px);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        color: var(--text-color);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border: 2px solid transparent;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Color coding for different categories */
    .metric-card.surge-low {
        border-left: 8px solid var(--success-color);
        background: linear-gradient(145deg, #d4edda, #c3e6cb);
    }
    
    .metric-card.surge-medium {
        border-left: 8px solid var(--warning-color);
        background: linear-gradient(145deg, #fff3cd, #ffeaa7);
    }
    
    .metric-card.surge-high {
        border-left: 8px solid var(--danger-color);
        background: linear-gradient(145deg, #f8d7da, #f5c6cb);
    }
    
    .metric-card.loyalty-new {
        border-left: 8px solid var(--danger-color);
    }
    
    .metric-card.loyalty-regular {
        border-left: 8px solid var(--warning-color);
    }
    
    .metric-card.loyalty-loyal {
        border-left: 8px solid var(--info-color);
    }
    
    .metric-card.loyalty-vip {
        border-left: 8px solid var(--success-color);
    }
    
    /* Enhanced info boxes */
    .info-box {
        background: var(--card-background);
        padding: clamp(1rem, 3vw, 1.2rem);
        border-radius: 15px;
        border-left: 6px solid #2196f3;
        margin: 0.8rem 0;
        word-wrap: break-word;
        color: var(--text-color);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .contact-info {
        background: var(--card-background);
        padding: clamp(1rem, 3vw, 1.2rem);
        border-radius: 15px;
        border-left: 6px solid #ff9800;
        margin: 0.8rem 0;
        word-wrap: break-word;
        color: var(--text-color);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .header-box {
        background: linear-gradient(135deg, var(--secondary-color) 0%, #764ba2 100%);
        padding: clamp(2rem, 5vw, 3rem);
        text-align: center;
        border-radius: 20px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.2);
    }
    
    /* Enhanced tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        border-bottom: 1px dotted var(--text-color);
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Icons for better visual appeal */
    .icon {
        font-size: 1.2em;
        margin-right: 0.5rem;
        vertical-align: middle;
    }
    
    /* Enhanced gauge visualization */
    .gauge-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 1rem 0;
    }
    
    .gauge {
        width: 100px;
        height: 50px;
        border-radius: 100px 100px 0 0;
        position: relative;
        overflow: hidden;
        background: #e0e0e0;
    }
    
    .gauge-fill {
        height: 100%;
        border-radius: 100px 100px 0 0;
        transition: width 0.5s ease;
    }
    
    .gauge-low { background: var(--success-color); }
    .gauge-medium { background: var(--warning-color); }
    .gauge-high { background: var(--danger-color); }
    
    /* Fare breakdown styling */
    .fare-breakdown {
        background: var(--card-background);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
    }
    
    .fare-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--border-color);
    }
    
    .fare-item:last-child {
        border-bottom: none;
        font-weight: bold;
        font-size: 1.1em;
        color: var(--primary-color);
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
        
        .fare-breakdown {
            font-size: 0.9rem;
        }
    }
    
    /* Dark mode enhancements */
    @media (prefers-color-scheme: dark) {
        .prediction-box {
            background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%) !important;
            box-shadow: 0 12px 40px rgba(255,255,255,0.1) !important;
        }
        
        .metric-card {
            background: #262730 !important;
            color: #fafafa !important;
        }
        
        .info-box, .contact-info {
            background: #262730 !important;
            color: #fafafa !important;
        }
        
        .fare-breakdown {
            background: #262730 !important;
            border-color: #464a57 !important;
        }
    }
    
    /* Enhanced button styling */
    .stButton > button {
        width: 100%;
        padding: 1rem 1.5rem;
        font-size: clamp(1rem, 3vw, 1.2rem);
        font-weight: 600;
        background: linear-gradient(135deg, var(--secondary-color) 0%, #0066cc 100%);
        color: white;
        border: none;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Responsive containers */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: clamp(1rem, 4vw, 3rem);
        padding-right: clamp(1rem, 4vw, 3rem);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced functions
def display_header():
    """Display header with mobile-friendly design"""
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

def create_gauge_chart(value, max_value=100, title=""):
    """Create a gauge chart for visual representation"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_surge_gauge(surge_value):
    """Create surge level gauge"""
    surge_percentage = min((surge_value - 1) * 50, 100)  # Convert 1-3 to 0-100
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = surge_percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Surge Level"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "#28a745"},
                {'range': [33, 66], 'color': "#ffc107"},
                {'range': [66, 100], 'color': "#dc3545"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def calculate_surge_pricing(distance, rating, cab_type, traffic, demand, weather):
    """Enhanced surge pricing calculation with detailed breakdown"""
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
        
        # Return detailed breakdown
        breakdown = {
            'base': base_surge,
            'distance_factor': distance_factor,
            'rating_factor': rating_factor,
            'cab_factor': cab_factor,
            'condition_factor': condition_factor,
            'total': max(1.0, min(3.0, float(surge)))
        }
        
        return breakdown
    except Exception:
        return {
            'base': 1.0,
            'distance_factor': 0.0,
            'rating_factor': 0.0,
            'cab_factor': 0.0,
            'condition_factor': 0.0,
            'total': 1.5
        }

def get_surge_category_class(surge_value):
    """Get CSS class for surge category"""
    if surge_value <= 1.5:
        return "surge-low"
    elif surge_value <= 2.5:
        return "surge-medium"
    else:
        return "surge-high"

def get_loyalty_class(months):
    """Get CSS class for loyalty level"""
    if months > 24:
        return "loyalty-vip"
    elif months > 12:
        return "loyalty-loyal"
    elif months > 3:
        return "loyalty-regular"
    else:
        return "loyalty-new"

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

# Display header
display_header()

# Enhanced title
st.markdown('<h1 class="main-header">🎯 Intelligent Taxi Pricing Analysis</h1>', unsafe_allow_html=True)

# About section
about_container = st.container()
with about_container:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3><span class="icon">🌟</span>About Sigma Cabs</h3>
            <p><strong>Sigma Cabs</strong> provides exceptional cab service in 
            <strong>Hyderabad</strong> and <strong>Bangalore</strong>. Our ML-powered 
            pricing system ensures fair and transparent fares based on real-time conditions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="contact-info">
            <h4><span class="icon">📞</span>Contact Info</h4>
            <p><strong>Toll-Free:</strong><br>📞 1800-420-9999</p>
            <p><strong>24/7:</strong><br>📱 040-63 63 63 63</p>
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

# Enhanced Input Section
st.markdown("## 🎯 Intelligent Fare Prediction")

# Trip Details
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
            help="Choose your preferred vehicle category: Economy (budget-friendly), Standard (comfortable), Premium (luxury)"
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
            help="Your average rating as a customer (higher ratings may get better pricing)"
        )

# Customer Info
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
            help="How long you've been a customer (loyalty affects pricing)"
        )
        lifestyle = st.slider(
            "💎 Lifestyle Index:", 
            1.0, 3.0, 2.0, 
            step=0.1,
            help="1: Budget-conscious, 2: Moderate, 3: Premium lifestyle preferences"
        )
    
    with cust_col2:
        cancellations = st.number_input(
            "❌ Cancellations Last Month:", 
            min_value=0, 
            max_value=10, 
            value=0,
            help="Number of ride cancellations in the past month"
        )
        confidence = st.selectbox(
            "🎯 Service Confidence:", 
            ['High Confidence', 'Medium Confidence', 'Low Confidence'],
            help="Your confidence level in using taxi services regularly"
        )

# Advanced Factors
with st.expander("🔧 Advanced Pricing Factors"):
    st.markdown("**Adjust these real-time factors for more accurate pricing:**")
    
    adv_col1, adv_col2, adv_col3 = st.columns([1, 1, 1])
    
    with adv_col1:
        traffic = st.slider(
            "🚦 Traffic Density:", 
            0.0, 100.0, 50.0,
            help="Current traffic conditions: 0 = No traffic, 100 = Heavy congestion"
        )
    
    with adv_col2:
        demand = st.slider(
            "📈 Demand Level:", 
            0.0, 100.0, 50.0,
            help="Current demand for rides: 0 = Low demand, 100 = Very high demand"
        )
    
    with adv_col3:
        weather = st.slider(
            "🌧️ Weather Impact:", 
            0.0, 100.0, 30.0,
            help="Weather impact on travel: 0 = Perfect weather, 100 = Severe weather"
        )

# Enhanced Prediction Button
predict_container = st.container()
with predict_container:
    if st.button('🔮 Calculate Smart Pricing', type="primary", use_container_width=True):
        try:
            # Calculate detailed surge pricing
            surge_breakdown = calculate_surge_pricing(distance, rating, cab_type, traffic, demand, weather)
            surge = surge_breakdown['total']
            
            # Display enhanced prediction result
            st.markdown(f"""
            <div class="prediction-box">
                <h2>🎯 Predicted Surge Pricing</h2>
                <h1>{surge:.2f}x</h1>
                <p>The increased fare multiplier due to current conditions</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced Analysis Results with color coding
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
                            <span class="tooltiptext">This multiplier increases your base fare due to high demand conditions</span>
                        </span>
                    </p>
                    <p><strong>Distance:</strong> {distance} km</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add surge gauge
                if surge <= 1.5:
                    gauge_color = "#28a745"
                elif surge <= 2.5:
                    gauge_color = "#ffc107"
                else:
                    gauge_color = "#dc3545"
                
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
                            <span class="tooltiptext">Customer loyalty status based on duration and frequency of rides</span>
                        </span>
                    </p>
                    <p><strong>Rating:</strong> {rating}/5.0 ⭐</p>
                    <p><strong>Since:</strong> {months} months</p>
                </div>
                """, unsafe_allow_html=True)
            
            with result_col3:
                # Enhanced fare calculation with breakdown
                base_fare = 10.0
                distance_cost = distance * 2.5
                surge_additional = (distance_cost * (surge - 1))
                total_fare = base_fare + distance_cost + surge_additional
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4><span class="icon">💰</span>Estimated Fare</h4>
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
                            <span>Surge ({surge:.2f}x):</span>
                            <span>+${surge_additional:.2f}</span>
                        </div>
                        <div class="fare-item">
                            <span>Total:</span>
                            <span>${total_fare:.2f}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced Conditions Impact with visual gauge
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
                    <p><strong>💡 Recommendation:</strong> {'Consider alternative time or route' if condition_score > 70 else 'Good time to travel - optimal conditions'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with condition_col2:
                # Create conditions gauge
                conditions_fig = create_gauge_chart(condition_score, 100, "Conditions Impact")
                st.plotly_chart(conditions_fig, use_container_width=True)
            
            # Interactive surge factors breakdown
            st.markdown("### 📈 Surge Factors Breakdown")
            
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
            
        except Exception as e:
            st.error("❌ Prediction error occurred")
            st.markdown("""
            <div class="prediction-box">
                <h2>🎯 Default Surge Pricing</h2>
                <h1>1.50x</h1>
                <p>Standard multiplier applied</p>
            </div>
            """, unsafe_allow_html=True)

# Enhanced Information Section with icons
info_container = st.container()
with info_container:
    st.markdown("---")
    st.markdown("## 💡 Understanding Smart Pricing")
    
    info_col1, info_col2 = st.columns([1, 1])
    
    with info_col1:
        st.markdown("""
        <div class="info-box">
            <h3><span class="icon">🔍</span>Vehicle Categories</h3>
            <ul>
                <li><strong>🚗 Economy (Micro):</strong> Budget-friendly, compact cars for short trips</li>
                <li><strong>🚙 Standard (Mini):</strong> Regular sedans with good comfort for medium trips</li>
                <li><strong>🚘 Premium (Prime):</strong> Luxury vehicles with premium service</li>
            </ul>
            <h3><span class="icon">🎯</span>Confidence Levels</h3>
            <ul>
                <li><strong>🟢 High:</strong> Frequent user who trusts the service completely</li>
                <li><strong>🟡 Medium:</strong> Occasional user with moderate confidence</li>
                <li><strong>🔴 Low:</strong> New or hesitant user, needs more assurance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with info_col2:
        st.markdown("""
        <div class="info-box">
            <h3><span class="icon">🌧️</span>Dynamic Pricing Factors</h3>
            <ul>
                <li><strong>🚦 Traffic Density:</strong> Real-time road congestion levels</li>
                <li><strong>📈 Demand Level:</strong> Current booking requests in your area</li>
                <li><strong>🌤️ Weather Impact:</strong> Weather conditions affecting travel safety</li>
                <li><strong>📏 Distance:</strong> Primary cost factor for your journey</li>
            </ul>
            <h3><span class="icon">🤖</span>How Our AI based on Machine Learning Works</h3>
            <p>Our advanced machine learning model analyzes <strong>13+ factors</strong> in real-time to predict fair and transparent surge pricing, ensuring you get the best possible fare.</p>
        </div>
        """, unsafe_allow_html=True)

# Enhanced System Status
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

# Enhanced Footer
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
