import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_features(df):
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract time features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Day'] = df['Date'].dt.day
    
    # Lag features
    df['Weekly_Sales_Lag1'] = df.groupby('Store')['Weekly_Sales'].shift(1)
    
    # Rolling features
    df['Weekly_Sales_Rolling_Mean'] = df.groupby('Store')['Weekly_Sales'].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean()
    )
    
    return df

def scale_features(df, features_to_scale):
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    return df