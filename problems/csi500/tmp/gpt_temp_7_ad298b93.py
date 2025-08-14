import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def heuristics_v2(df, sentiment_data):
    # Calculate Daily Price Change
    df['Daily_Price_Change'] = df['close'].diff()
    
    # Compute Volume Adjusted Return
    df['Volume_Adjusted_Return'] = df['Daily_Price_Change'] / df['volume']
    
    # Initialize EMA value with Volume Adjusted Return of day 1
    df['EMA_Volume_Adjusted_Return'] = df['Volume_Adjusted_Return'].copy()
    
    # Calculate Adaptive EMA of Volume Adjusted Returns
    for i in range(1, len(df)):
        N = determine_ema_days(sentiment_data.iloc[i-1])
        multiplier = 2 / (N + 1)
        df.loc[df.index[i], 'EMA_Volume_Adjusted_Return'] = (df.loc[df.index[i], 'Volume_Adjusted_Return'] * multiplier) + (df.loc[df.index[i-1], 'EMA_Volume_Adjusted_Return'] * (1 - multiplier))
    
    # Calculate High-Low Range
    df['High_Low_Range'] = df['high'] - df['low']
    
    # Calculate Volume Weighted High-Low Range
    df['Volume_Weighted_High_Low_Range'] = df['High_Low_Range'] * df['volume']
    
    # Initialize EMA value with Volume Weighted High-Low Range of day 1
    df['EMA_Volume_Weighted_High_Low_Range'] = df['Volume_Weighted_High_Low_Range'].copy()
    
    # Calculate Adaptive EMA of Volume Weighted High-Low Range
    for i in range(1, len(df)):
        N = determine_ema_days(sentiment_data.iloc[i-1])
        multiplier = 2 / (N + 1)
        df.loc[df.index[i], 'EMA_Volume_Weighted_High_Low_Range'] = (df.loc[df.index[i], 'Volume_Weighted_High_Low_Range'] * multiplier) + (df.loc[df.index[i-1], 'EMA_Volume_Weighted_High_Low_Range'] * (1 - multiplier))
    
    # Integrate Market Sentiment
    df['Sentiment_Score'] = sentiment_data['sentiment_score']
    
    # Dynamic Factor Adjustment using a machine learning model
    features = ['EMA_Volume_Adjusted_Return', 'EMA_Volume_Weighted_High_Low_Range', 'Sentiment_Score']
    X = df[features].shift(1).dropna()
    y = df['close'].pct_change().shift(-1).dropna()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    df['Predicted_Adjusted_Alpha_Factor'] = model.predict(df[features])
    
    # Generate Final Alpha Factor
    final_alpha_factor = df['Predicted_Adjusted_Alpha_Factor']
    
    return final_alpha_factor.dropna()

def determine_ema_days(sentiment):
    # Example: Determine the number of days in EMA based on market sentiment
    if sentiment < -0.5:
        return 30
    elif sentiment > 0.5:
        return 14
    else:
        return 20
