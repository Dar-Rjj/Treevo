import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def heuristics_v2(df, sentiment_data=None):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Volatility Calculation
    df['volatility'] = df[['high', 'low', 'close']].rolling(window=20).std().mean(axis=1)
    
    # Adaptive Window Size
    def adaptive_window_size(volatility):
        if volatility > df['volatility'].quantile(0.75):
            return 5
        else:
            return 30
    
    df['window_size'] = df['volatility'].apply(adaptive_window_size)
    
    # Incorporate Liquidity
    df['average_volume'] = df['volume'].rolling(window=20).mean()
    df['liquidity_factor'] = df['average_volume'].apply(lambda x: 1 + (x / df['average_volume'].mean()) if x > df['average_volume'].mean() else 1 - (x / df['average_volume'].mean()))
    df['adjusted_volume_weighted_return'] = df['volume_weighted_return'] * df['liquidity_factor']
    
    # Incorporate Sentiment
    if sentiment_data is not None:
        df = df.merge(sentiment_data, left_index=True, right_index=True, how='left')
        df['sentiment_adjusted_return'] = df['adjusted_volume_weighted_return'] * (1 + df['sentiment'] * 0.1)
    else:
        df['sentiment_adjusted_return'] = df['adjusted_volume_weighted_return']
    
    # Machine Learning for Adaptive Window Optimization
    X = df[['volatility', 'average_volume', 'sentiment']].dropna()
    y = df['window_size'].reindex(X.index)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    df['predicted_window_size'] = model.predict(df[['volatility', 'average_volume', 'sentiment']])
    
    # Calculate Rolling Statistics with Predicted Window Size
    df['rolling_mean'] = df.groupby('predicted_window_size')['sentiment_adjusted_return'].transform(lambda x: x.rolling(window=int(x.name)).mean())
    df['rolling_std'] = df.groupby('predicted_window_size')['sentiment_adjusted_return'].transform(lambda x: x.rolling(window=int(x.name)).std())
    
    # Final Alpha Factor
    df['alpha_factor'] = df['rolling_mean'] / df['rolling_std']
    
    return df['alpha_factor']
