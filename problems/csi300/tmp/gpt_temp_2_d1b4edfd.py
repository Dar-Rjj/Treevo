import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Short-term momentum (5-day) normalized by recent volatility (5-day ATR)
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    atr_5d = ((df['high'] - df['low']).rolling(window=5).mean() + 
              (df['high'] - df['close'].shift()).abs().rolling(window=5).mean() + 
              (df['low'] - df['close'].shift()).abs().rolling(window=5).mean()) / 3
    volatility_normalized_momentum = momentum_5d / (atr_5d + 1e-7)
    
    # Volume acceleration divergence: current vs 3-day average vs 10-day trend
    volume_3d_avg = df['volume'].rolling(window=3).mean()
    volume_10d_trend = df['volume'].rolling(window=10).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    volume_acceleration_divergence = (df['volume'] / volume_3d_avg - 1) - volume_10d_trend
    
    # Intraday positioning strength: close relative to daily range over 7-day window
    daily_range_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    intraday_position_strength = daily_range_position.rolling(window=7).mean()
    
    # Combined alpha: volatility-normalized momentum amplified by volume divergence and intraday positioning
    alpha_factor = (volatility_normalized_momentum * 
                   (1 + volume_acceleration_divergence) * 
                   (1 + intraday_position_strength - 0.5))
    
    return alpha_factor
