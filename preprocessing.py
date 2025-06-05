# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_input_data(data_dict):
    """
    Preprocess input data untuk prediction
    """
    # Convert dict to DataFrame
    df = pd.DataFrame([data_dict])
    
    # Feature engineering yang sama seperti training
    df['Customer_Loyalty_Segment'] = pd.cut(df['Customer_Since_Months'], 
                                          bins=[0, 3, 12, 24, float('inf')],
                                          labels=['New', 'Regular', 'Loyal', 'VIP'])
    
    df['Distance_Rating_Interaction'] = df['Trip_Distance'] * df['Customer_Rating']
    df['Service_Quality_Score'] = (df['Customer_Rating'] * 0.6 + 
                                  (5 - df['Cancellation_Last_1Month'].clip(0, 5)) * 0.4)
    
    # Risk score
    df['Risk_Score'] = np.where(df['Cancellation_Last_1Month'] > 2, 'High',
                               np.where(df['Cancellation_Last_1Month'] > 0, 'Medium', 'Low'))
    
    return df

def encode_features(df):
    """
    Encode categorical features
    """
    # One-hot encoding untuk categorical features
    categorical_features = ['Destination_Type', 'Gender', 'Customer_Loyalty_Segment', 'Risk_Score']
    
    for feature in categorical_features:
        if feature in df.columns:
            dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
    
    # Label encoding untuk ordinal features
    if 'Type_of_Cab' in df.columns:
        cab_mapping = {'Micro': 0, 'Mini': 1, 'Prime': 2, 'XL': 3}
        df['Type_of_Cab_encoded'] = df['Type_of_Cab'].map(cab_mapping)
    
    if 'Confidence_Life_Style_Index' in df.columns:
        confidence_mapping = {'A': 3, 'B': 2, 'C': 1}
        df['Confidence_Life_Style_Index_encoded'] = df['Confidence_Life_Style_Index'].map(confidence_mapping)
    
    return df
