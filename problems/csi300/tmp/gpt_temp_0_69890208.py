import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Volatility-normalized momentum with volume-weighted signals using robust rolling measures
    # Focus on price efficiency via range normalization and volume confirmation
    
    # Calculate momentum using close-to-close returns
    returns = df['close'].pct_change()
    
    # Robust volatility measure using median absolute deviation over 20-day window
    volatility = returns.rolling(window=20, min_periods=10).apply(lambda x: np.median(np.abs(x - np.median(x))))
    volatility_adj = volatility + 1e-7
    
    # Volume efficiency signal - ratio of current volume to rolling median
    volume_median = df['volume'].rolling(window=10, min_periods=5).median()
    volume_efficiency = df['volume'] / (volume_median + 1e-7)
    
    # Price range efficiency - normalized daily range
    daily_range = (df['high'] - df['low']) / df['close']
    range_efficiency = daily_range.rolling(window=5, min_periods=3).mean()
    
    # Combine signals: volatility-normalized momentum weighted by volume and range efficiency
    factor = (returns / volatility_adj) * volume_efficiency * (1 - range_efficiency)
    
    return factor
