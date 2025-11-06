import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum-volume alpha with volatility regime adaptation.
    
    This factor blends multi-timeframe momentum signals with volume acceleration,
    adapts to volatility regimes using exponential smoothing, and emphasizes
    recent data while normalizing by short-term volatility for robust signals.
    """
    # Hierarchical momentum across multiple timeframes
    momentum_1d = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Exponential smoothing with recent emphasis (shorter spans for recent data)
    momentum_1d_smooth = momentum_1d.ewm(span=2).mean()
    momentum_3d_smooth = momentum_3d.ewm(span=4).mean()
    momentum_5d_smooth = momentum_5d.ewm(span=6).mean()
    
    # Weighted hierarchical momentum blend (heavier weights for shorter timeframes)
    hierarchical_momentum = 0.5 * momentum_1d_smooth + 0.3 * momentum_3d_smooth + 0.2 * momentum_5d_smooth
    
    # Volume acceleration across multiple windows
    volume_current = df['volume']
    volume_3d_avg = df['volume'].rolling(window=3).mean()
    volume_5d_avg = df['volume'].rolling(window=5).mean()
    volume_10d_avg = df['volume'].rolling(window=10).mean()
    
    # Volume acceleration ratios with recent emphasis
    volume_accel_3d = volume_current / (volume_3d_avg + 1e-7)
    volume_accel_5d = volume_current / (volume_5d_avg + 1e-7)
    volume_accel_10d = volume_current / (volume_10d_avg + 1e-7)
    
    # Weighted volume acceleration blend
    volume_acceleration = 0.6 * volume_accel_3d + 0.3 * volume_accel_5d + 0.1 * volume_accel_10d
    
    # Short-term volatility regime (5-day rolling standard deviation)
    returns_1d = df['close'].pct_change()
    short_term_vol = returns_1d.rolling(window=5).std()
    
    # Volatility-normalized momentum component
    volatility_normalized_momentum = hierarchical_momentum / (short_term_vol + 1e-7)
    
    # Volume persistence confirmation (volume trend consistency)
    volume_trend_3d = df['volume'].rolling(window=3).apply(lambda x: (x[-1] > x[0]).astype(float))
    volume_trend_5d = df['volume'].rolling(window=5).apply(lambda x: (x[-1] > x[0]).astype(float))
    volume_persistence = 0.7 * volume_trend_3d + 0.3 * volume_trend_5d
    
    # Core factor: volatility-normalized momentum amplified by volume acceleration
    core_factor = volatility_normalized_momentum * volume_acceleration
    
    # Apply volume persistence as confirmation filter
    alpha_factor = core_factor * volume_persistence
    
    # Final exponential smoothing for regime adaptation
    alpha_factor_smooth = alpha_factor.ewm(span=3).mean()
    
    return alpha_factor_smooth
