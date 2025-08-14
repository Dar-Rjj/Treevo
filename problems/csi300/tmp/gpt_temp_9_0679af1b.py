import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Volume-Weighted Price Returns
    df['open_t1'] = df['open'].shift(-1)
    df['simple_returns'] = (df['open_t1'] - df['close']) / df['close']
    df['volume_weighted_returns'] = df['simple_returns'] * df['volume']
    
    # Identify Volume Surge Days
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['rolling_volume_mean'] = df['volume'].rolling(window=5).mean()
    df['is_surge_day'] = (df['volume'] > df['rolling_volume_mean']).astype(bool)
    
    # Calculate Adaptive Volatility
    df['daily_returns'] = df['close'].pct_change()
    recent_volatility = df['daily_returns'].rolling(window=20).std().dropna().iloc[-1]
    lookback_period = max(20, min(60, 20 + int(40 * recent_volatility)))
    df['volatility'] = df['daily_returns'].rolling(window=lookback_period).std()
    
    # Adjust for Volume Trends
    df['volume_moving_avg'] = df['volume'].rolling(window=20).mean()
    df['volume_std'] = df['volume'].rolling(window=20).std()
    df['volume_z_score'] = (df['volume'] - df['volume_moving_avg']) / df['volume_std']
    df['adjusted_volatility'] = df['volatility'] * (1 + abs(df['volume_z_score']))
    
    # Refine Surge Factors
    df['volume_surge_ratio'] = df['volume'] / df['volume'].shift(1)
    surge_factor_conditions = [
        (df['volume_surge_ratio'] > 2.5, 1.8),
        ((df['volume_surge_ratio'] > 2.0) & (df['volume_surge_ratio'] <= 2.5), 1.5),
        ((df['volume_surge_ratio'] > 1.5) & (df['volume_surge_ratio'] <= 2.0), 1.2),
        (True, 1.0)
    ]
    df['refined_surge_factor'] = pd.np.select(
        [condition[0] for condition in surge_factor_conditions],
        [condition[1] for condition in surge_factor_conditions],
        default=1.0
    )
    
    # Adjust Volume-Weighted Returns by Adaptive Volatility
    df['adjusted_returns'] = df['volume_weighted_returns'] / df['adjusted_volatility']
    
    # Combine Adjusted Returns with Refined Volume Surge Indicator
    df.loc[df['is_surge_day'], 'final_factor'] = df['adjusted_returns'] * df['refined_surge_factor']
    df.loc[~df['is_surge_day'], 'final_factor'] = df['adjusted_returns']
    
    return df['final_factor']
