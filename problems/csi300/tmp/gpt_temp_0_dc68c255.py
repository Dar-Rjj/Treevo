import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum-volume alpha with volatility regime adjustment.
    
    This factor blends multiple momentum timeframes with volume acceleration signals,
    applies exponential smoothing for recent data emphasis, normalizes by short-term
    volatility, and confirms with volume persistence for robust predictive signals.
    """
    # Hierarchical momentum across multiple timeframes
    momentum_1d = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Exponentially smoothed momentum with recent emphasis
    momentum_smooth_1d = momentum_1d.ewm(span=3).mean()
    momentum_smooth_3d = momentum_3d.ewm(span=5).mean()
    momentum_smooth_5d = momentum_5d.ewm(span=8).mean()
    
    # Hierarchical momentum blend with decaying weights
    momentum_hierarchy = (0.5 * momentum_smooth_1d + 
                         0.3 * momentum_smooth_3d + 
                         0.2 * momentum_smooth_5d)
    
    # Volume acceleration: rate of change in volume
    volume_accel_1d = df['volume'] / (df['volume'].shift(1) + 1e-7) - 1
    volume_accel_3d = df['volume'] / (df['volume'].shift(3) + 1e-7) - 1
    
    # Exponentially smoothed volume acceleration
    volume_accel_smooth = (0.7 * volume_accel_1d.ewm(span=3).mean() + 
                          0.3 * volume_accel_3d.ewm(span=5).mean())
    
    # Volume persistence: consistency of volume trends
    volume_trend_5d = df['volume'].rolling(window=5).apply(
        lambda x: 1 if (x.diff().dropna() > 0).all() else 
                 (-1 if (x.diff().dropna() < 0).all() else 0), 
        raw=False
    )
    
    # Volatility regime: short-term volatility normalization
    returns_5d = df['close'].pct_change(periods=5)
    volatility_5d = returns_5d.rolling(window=10).std()
    
    # Core factor: momentum amplified by volume acceleration
    momentum_volume_core = momentum_hierarchy * volume_accel_smooth
    
    # Apply volume persistence confirmation
    volume_confirmed = momentum_volume_core * (1 + 0.2 * volume_trend_5d)
    
    # Normalize by volatility regime
    volatility_normalized = volume_confirmed / (volatility_5d + 1e-7)
    
    # Final exponential smoothing for signal refinement
    alpha_factor = volatility_normalized.ewm(span=5).mean()
    
    return alpha_factor
