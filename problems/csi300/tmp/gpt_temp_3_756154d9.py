import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe momentum blend: 5-day and 10-day weighted combination
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_10d = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    momentum_blend = 0.6 * momentum_5d + 0.4 * momentum_10d
    
    # Volatility-scaled volume: volume normalized by 20-day price volatility
    returns_1d = df['close'].pct_change()
    vol_20d = returns_1d.rolling(window=20, min_periods=10).std()
    volume_scaled = df['volume'] / (vol_20d + 1e-7)
    
    # Quantile-based regime detection: identify high/low volatility periods
    vol_regime = vol_20d.rolling(window=50, min_periods=25).apply(
        lambda x: 1 if x.iloc[-1] > x.quantile(0.7) else (-1 if x.iloc[-1] < x.quantile(0.3) else 0),
        raw=False
    )
    
    # Range efficiency with regime adjustment
    daily_range = df['high'] - df['low']
    range_efficiency = np.abs(df['close'] - df['close'].shift(1)) / (daily_range + 1e-7)
    range_efficiency_adj = range_efficiency * (1 + 0.2 * vol_regime)
    
    # Multiplicative combination with clear component separation
    alpha_factor = momentum_blend * volume_scaled * range_efficiency_adj
    
    return alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
