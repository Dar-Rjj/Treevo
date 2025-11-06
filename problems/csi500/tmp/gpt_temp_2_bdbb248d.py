import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Stable volatility scaling with normalized volume-price divergence
    # Multiplies intraday mean reversion by range efficiency
    
    # Intraday mean reversion: (close - open) / (high - low)
    # Captures daily price reversal relative to daily range
    intraday_mean_reversion = (df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Range efficiency: (close - open) / (high - low) absolute value
    # Measures how efficiently price moves from open to close within daily range
    range_efficiency = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Volume-price divergence: volume normalized by 10-day median
    # Uses median for more stable normalization against outliers
    volume_normalized = df['volume'] / df['volume'].rolling(window=10, min_periods=5).median()
    
    # Price divergence: current close vs 5-day median price
    # Captures deviation from medium-term equilibrium
    price_divergence = (df['close'] - df['close'].rolling(window=5, min_periods=3).median()) / df['close'].rolling(window=5, min_periods=3).median()
    
    # Stable volatility: 20-day median absolute deviation of returns
    # More robust volatility measure than standard deviation
    returns = df['close'] / df['close'].shift(1) - 1
    stable_volatility = returns.rolling(window=20, min_periods=10).apply(lambda x: np.median(np.abs(x - np.median(x))))
    
    # Combine factors: intraday mean reversion amplified by range efficiency,
    # multiplied by volume-price divergence, normalized by stable volatility
    alpha_factor = (intraday_mean_reversion * range_efficiency * volume_normalized * price_divergence) / stable_volatility
    
    return alpha_factor
