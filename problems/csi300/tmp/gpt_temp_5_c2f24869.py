import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    amount = df['amount']
    
    intraday_momentum = (close - (high + low) / 2) / (high - low + 1e-8)
    volume_efficiency = np.where(volume > 0, amount / volume, 0)
    combined_strength = intraday_momentum * volume_efficiency
    
    high_low_volatility = (high - low) / (close.rolling(window=5).mean() + 1e-8)
    price_trend = close / close.rolling(window=10).mean() - 1
    
    volatility_trend_ratio = high_low_volatility / (price_trend.abs() + 1e-8)
    
    raw_alpha = combined_strength - volatility_trend_ratio
    heuristics_matrix = pd.Series(raw_alpha, index=df.index)
    
    return heuristics_matrix
