import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['daily_dollar_value'] = df['close'] * df['volume']
    total_volume = df['volume'].sum()
    total_dollar_value = df['daily_dollar_value'].sum()
    vwap = total_dollar_value / total_volume
    
    # Calculate VWAP Deviation
    df['vwap_deviation'] = df['close'] - vwap
    
    # Calculate Cumulative VWAP Deviation
    df['cumulative_vwap_deviation'] = df['vwap_deviation'].cumsum()
    
    # Integrate Volatility
    df['true_range'] = df['high'] - df['low']
    df['volatility_20_ma'] = df['true_range'].rolling(window=20).mean()
    df['normalized_volatility'] = df['true_range'] / df['volatility_20_ma']
    
    # Incorporate Market Sentiment
    df['daily_change'] = df['close'].diff()
    df['positive_changes'] = df['daily_change'].apply(lambda x: x if x > 0 else 0)
    df['negative_changes'] = df['daily_change'].apply(lambda x: x if x < 0 else 0)
    df['pos_sum_10'] = df['positive_changes'].rolling(window=10).sum()
    df['neg_sum_10'] = df['negative_changes'].rolling(window=10).sum()
    df['sentiment_score'] = (df['pos_sum_10'] - df['neg_sum_10']) / (df['pos_sum_10'] + df['neg_sum_10'])
    
    # Prepare Features
    features = df[['cumulative_vwap_deviation', 'normalized_volatility', 'sentiment_score']].dropna()
    
    # Train Machine Learning Model
    X = features.values
    y = df['close'].shift(-1).fillna(method='ffill').loc[features.index].values
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predict Weights
    predicted_weights = model.predict(X)
    
    # Final Alpha Signal
    alpha_signal = (features * predicted_weights[:, None]).sum(axis=1)
    
    return alpha_signal
