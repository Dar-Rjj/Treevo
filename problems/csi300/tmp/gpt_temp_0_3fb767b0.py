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
    df['volume_change'] = df['volume'].pct_change()
    df['rolling_vol_avg'] = df['volume'].rolling(window=5).mean()
    df['volume_surge'] = df['volume'] > df['rolling_vol_avg']
    
    # Calculate Adaptive Volatility
    df['daily_returns'] = df['close'].pct_change()
    def dynamic_lookback_volatility(returns, min_days=20, max_days=60):
        recent_volatility = returns.rolling(window=min_days).std().iloc[-1]
        if recent_volatility < 0.01:
            lookback = max_days
        elif recent_volatility > 0.05:
            lookback = min_days
        else:
            lookback = int((max_days - min_days) * (0.05 - recent_volatility) / 0.04 + min_days)
        return returns.rolling(window=lookback).std()
    
    df['adaptive_volatility'] = dynamic_lookback_volatility(df['daily_returns'])
    df['vol_moving_avg'] = df['volume'].rolling(window=20).mean()
    df['vol_std'] = df['volume'].rolling(window=20).std()
    df['vol_z_score'] = (df['volume'] - df['vol_moving_avg']) / df['vol_std']
    df['adjusted_volatility'] = df['adaptive_volatility'] * (1 + np.abs(df['vol_z_score']))
    
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
    df['adjusted_volume_weighted_returns'] = df['volume_weighted_returns'] / df['adjusted_volatility']
    
    # Combine Adjusted Returns with Refined Volume Surge Indicator
    df['final_factor'] = df['adjusted_volume_weighted_returns']
    df.loc[df['volume_surge'], 'final_factor'] *= df['refined_surge_factor']
    
    return df['final_factor'].dropna()

# Example usage
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# factor_series = heuristics_v2(df)
