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
    df['daily_volume_change'] = df['volume'] - df['volume'].shift(1)
    df['volume_rolling_mean'] = df['volume'].rolling(window=5).mean()
    df['is_surge_day'] = df['volume'] > df['volume_rolling_mean']
    
    # Calculate Adaptive Volatility
    df['daily_returns'] = df['close'].pct_change()
    df['volatility'] = df['daily_returns'].rolling(window=20).std()  # Initial lookback
    df['dynamic_volatility'] = df['volatility'].rolling(window=20).apply(
        lambda x: np.std(x) * (1 + np.abs((x[-1] - x.mean()) / np.std(x))), raw=False
    )
    
    # Adjust for Volume Trends
    df['volume_moving_avg'] = df['volume'].rolling(window=20).mean()
    df['volume_std'] = df['volume'].rolling(window=20).std()
    df['volume_z_score'] = (df['volume'] - df['volume_moving_avg']) / df['volume_std']
    df['adjusted_volatility'] = df['dynamic_volatility'] * (1 + np.abs(df['volume_z_score']))
    
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
    df['final_factor'] = df['adjusted_returns']
    df.loc[df['is_surge_day'], 'final_factor'] *= df['refined_surge_factor']
    
    return df['final_factor']

# Example usage:
# factor_values = heuristics_v2(market_data_df)
# print(factor_values)
