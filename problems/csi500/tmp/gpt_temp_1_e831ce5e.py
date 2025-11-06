import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # 3-7 day momentum reversals blended with 5-day volume acceleration, normalized by 20-day true range
    # Volatility-threshold smoothing adapts to different market regimes
    # Multiplicative synergy among momentum, volume, and volatility components for enhanced stability
    # Higher values indicate favorable momentum reversal patterns with supportive volume dynamics
    
    # 3-day and 7-day momentum for reversal detection
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    momentum_7d = (df['close'] - df['close'].shift(7)) / df['close'].shift(7)
    
    # Momentum reversal blend: shorter momentum relative to longer momentum
    momentum_reversal = momentum_3d - momentum_7d
    
    # 5-day volume acceleration: current volume vs 5-day average trend
    volume_5d_avg = df['volume'].rolling(window=5).mean()
    volume_acceleration = df['volume'] / volume_5d_avg
    
    # 20-day true range for volatility normalization
    true_range = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    volatility_20d = true_range.rolling(window=20).mean()
    
    # Volatility-threshold smoothing: regime adaptation
    volatility_threshold = volatility_20d.rolling(window=20).quantile(0.5)
    high_vol_regime = volatility_20d > volatility_threshold
    
    # Base factor: momentum reversal blended with volume acceleration, normalized by volatility
    base_factor = momentum_reversal * volume_acceleration / (volatility_20d + 1e-7)
    
    # Apply volatility-threshold smoothing: different smoothing for high vs low volatility regimes
    # In high volatility, use longer smoothing (5-day) for stability
    # In low volatility, use shorter smoothing (3-day) for responsiveness
    smooth_window = pd.Series(3, index=df.index)
    smooth_window[high_vol_regime] = 5
    
    # Create smoothed factor using dynamic window sizes
    smoothed_factor = base_factor.copy()
    for i in range(len(df)):
        if i >= smooth_window.iloc[i]:
            window_size = int(smooth_window.iloc[i])
            smoothed_factor.iloc[i] = base_factor.iloc[i-window_size+1:i+1].mean()
    
    return smoothed_factor
