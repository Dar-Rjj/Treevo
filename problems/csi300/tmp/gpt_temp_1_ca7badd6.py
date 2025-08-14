import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Volume-Weighted Price Returns
    df['next_open'] = df['open'].shift(-1)
    df['simple_returns'] = (df['next_open'] - df['close']) / df['close']
    df['volume_weighted_returns'] = df['simple_returns'] * df['volume']
    
    # Identify Volume Surge Days
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['rolling_avg_volume'] = df['volume'].rolling(window=5).mean()
    df['is_volume_surge'] = df['volume'] > df['rolling_avg_volume']
    
    # Calculate Adaptive Volatility
    df['daily_returns'] = df['close'].pct_change()
    def dynamic_lookback_volatility(x, min_days=20, max_days=60):
        recent_vol = x.rolling(window=min_days).std().iloc[-1]
        if recent_vol < 0.01:
            lookback = max_days
        elif recent_vol < 0.02:
            lookback = 40
        else:
            lookback = min_days
        return x.rolling(window=lookback).std()
    
    df['adaptive_volatility'] = df['daily_returns'].rolling(window=60).apply(dynamic_lookback_volatility, raw=False)
    
    # Adjust for Volume Trends
    df['vol_ma_20'] = df['volume'].rolling(window=20).mean()
    df['vol_std'] = df['volume'].rolling(window=20).std()
    df['vol_z_score'] = (df['volume'] - df['vol_ma_20']) / df['vol_std']
    df['adjusted_volatility'] = df['adaptive_volatility'] * (1 + abs(df['vol_z_score']))
    
    # Refine Surge Factors
    df['volume_surge_ratio'] = df['volume'] / df['volume'].shift(1)
    def refine_surge_factor(ratio):
        if ratio > 2.5:
            return 1.8
        elif ratio > 2.0:
            return 1.5
        elif ratio > 1.5:
            return 1.2
        else:
            return 1.0
    df['refined_surge_factor'] = df['volume_surge_ratio'].apply(refine_surge_factor)
    
    # Adjust Volume-Weighted Returns by Adaptive Volatility
    df['adjusted_returns'] = df['volume_weighted_returns'] / df['adjusted_volatility']
    
    # Combine Adjusted Returns with Refined Volume Surge Indicator
    df['alpha_factor'] = df['adjusted_returns']
    df.loc[df['is_volume_surge'], 'alpha_factor'] *= df['refined_surge_factor']
    
    return df['alpha_factor']

# Example usage
# df = pd.DataFrame({
#     'date': pd.date_range(start='2023-01-01', periods=100),
#     'open': np.random.rand(100) * 100,
#     'high': np.random.rand(100) * 100,
#     'low': np.random.rand(100) * 100,
#     'close': np.random.rand(100) * 100,
#     'amount': np.random.rand(100) * 100,
#     'volume': np.random.randint(100, 1000, size=100)
# })
# alpha_factor = heuristics_v2(df.set_index('date'))
# print(alpha_factor)
