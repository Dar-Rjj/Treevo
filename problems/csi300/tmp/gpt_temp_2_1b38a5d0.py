import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adjusted Volume Acceleration factor
    Combines volume acceleration with volatility regime adjustments
    """
    # Volume acceleration factor
    vol_5d_mean = df['volume'].rolling(window=5, min_periods=3).mean()
    vol_20d_mean = df['volume'].rolling(window=20, min_periods=10).mean()
    
    # Volume change rates
    vol_change_short = df['volume'] / vol_5d_mean
    vol_change_medium = vol_5d_mean / vol_20d_mean
    
    # Volume acceleration magnitude
    vol_acceleration = vol_change_short - vol_change_medium.shift(5)
    
    # Volatility scaling component
    # 5-day high-low range average
    daily_range = (df['high'] - df['low']) / df['close']
    range_5d_avg = daily_range.rolling(window=5, min_periods=3).mean()
    
    # 10-day close-to-close volatility
    returns = df['close'].pct_change()
    vol_10d = returns.rolling(window=10, min_periods=5).std()
    
    # Volatility regime adjustment
    vol_ratio = range_5d_avg / vol_10d
    vol_trend = vol_10d.rolling(window=5, min_periods=3).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1 if x.iloc[-1] < x.iloc[0] else 0
    )
    
    # Volatility scaling factor
    volatility_scaling = vol_ratio * (1 + 0.1 * vol_trend)
    
    # Directional momentum overlay
    price_position = (df['close'] - df['low']) / (df['high'] - df['low'])
    price_trend = df['close'].rolling(window=5, min_periods=3).apply(
        lambda x: 1 if x.iloc[-1] > x.mean() else -1
    )
    
    # Composite alpha factor
    alpha_factor = vol_acceleration * volatility_scaling * price_trend * price_position
    
    # Clean and normalize
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=20, min_periods=10).mean()) / alpha_factor.rolling(window=20, min_periods=10).std()
    
    return alpha_factor
