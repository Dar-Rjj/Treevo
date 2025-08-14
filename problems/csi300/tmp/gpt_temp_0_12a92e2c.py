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
    df['volume_rolling_mean'] = df['volume'].rolling(window=5).mean()
    df['is_volume_surge'] = df['volume'] > df['volume_rolling_mean']

    # Calculate Adaptive Volatility
    df['daily_returns'] = df['close'].pct_change()
    rolling_std = df['daily_returns'].rolling(window=20).std()
    recent_volatility = df['daily_returns'].rolling(window=10).std()
    lookback_period = 20 + (recent_volatility / rolling_std) * 40  # Adjust N based on recent volatility
    lookback_period = lookback_period.clip(lower=20, upper=60)
    df['adaptive_volatility'] = df['daily_returns'].rolling(window=lookback_period.astype(int)).std()

    # Adjust for Volume Trends
    volume_moving_avg = df['volume'].rolling(window=20).mean()
    volume_z_score = (df['volume'] - volume_moving_avg) / df['volume'].rolling(window=20).std()
    df['adjusted_volatility'] = df['adaptive_volatility'] * (1 + abs(volume_z_score))

    # Refine Surge Factors
    df['volume_surge_ratio'] = df['volume'] / df['volume'].shift(1)
    surge_factors = [2.0, 1.8, 1.5, 1.2]
    conditions = [
        df['volume_surge_ratio'] > 3.0,
        (df['volume_surge_ratio'] > 2.5) & (df['volume_surge_ratio'] <= 3.0),
        (df['volume_surge_ratio'] > 2.0) & (df['volume_surge_ratio'] <= 2.5),
        (df['volume_surge_ratio'] > 1.5) & (df['volume_surge_ratio'] <= 2.0)
    ]
    df['refined_surge_factor'] = pd.np.select(conditions, surge_factors, default=1.0)

    # Adjust Volume-Weighted Returns by Adaptive Volatility
    df['adjusted_returns'] = df['volume_weighted_returns'] / df['adjusted_volatility']

    # Combine Adjusted Returns with Refined Volume Surge Indicator
    df['final_factor'] = df['adjusted_returns'] * df['refined_surge_factor'] * df['is_volume_surge']

    return df['final_factor'].dropna()
