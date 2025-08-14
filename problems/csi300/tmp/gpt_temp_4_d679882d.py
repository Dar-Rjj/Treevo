import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

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
    def dynamic_volatility_lookback(std, n_max=60, n_min=20):
        recent_vol = std.rolling(window=n_min).std().iloc[-1]
        if recent_vol < 0.01:
            return n_max
        else:
            return max(n_min, min(int(1 / recent_vol), n_max))
    lookback = dynamic_volatility_lookback(df['daily_returns'])
    df['volatility'] = df['daily_returns'].rolling(window=lookback).std()
    df['volume_moving_avg'] = df['volume'].rolling(window=20).mean()
    df['volume_std'] = df['volume'].rolling(window=20).std()
    df['volume_z_score'] = (df['volume'] - df['volume_moving_avg']) / df['volume_std']
    df['adjusted_volatility'] = df['volatility'] * (1 + np.abs(df['volume_z_score']))
    
    # Refine Volume Surge Factors
    df['volume_surge_ratio'] = df['volume'] / df['volume'].shift(1)
    conditions = [
        (df['volume_surge_ratio'] > 2.5),
        (df['volume_surge_ratio'] > 2.0) & (df['volume_surge_ratio'] <= 2.5),
        (df['volume_surge_ratio'] > 1.5) & (df['volume_surge_ratio'] <= 2.0)
    ]
    choices = [1.8, 1.5, 1.2]
    df['surge_factor'] = np.select(conditions, choices, default=1.0)
    
    # Adjust Volume-Weighted Returns by Adaptive Volatility
    df['adjusted_returns'] = df['volume_weighted_returns'] / df['adjusted_volatility']
    
    # Combine Adjusted Returns with Refined Volume Surge Indicator
    df['final_factor'] = df['adjusted_returns'] * df['is_volume_surge'].astype(int) * df['surge_factor'] + df['adjusted_returns'] * (1 - df['is_volume_surge'].astype(int))
    
    return df['final_factor'].dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=True, index_col='date')
# alpha_factor = heuristics_v2(df)
