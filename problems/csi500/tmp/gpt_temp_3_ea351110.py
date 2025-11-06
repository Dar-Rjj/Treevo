import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe momentum blend with volatility normalization and volume acceleration
    # Combines 3-7 day momentum signals for robust trend detection
    # Normalizes by 20-day volatility for risk-adjusted positioning
    # Aligns with 5-day volume acceleration for confirmation
    # Uses regime-aware smoothing for signal stability
    
    # 3-7 day momentum blend (weighted combination)
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_7d = (df['close'] - df['close'].shift(7)) / df['close'].shift(7)
    
    # Weighted momentum blend (3d: 0.4, 5d: 0.35, 7d: 0.25)
    momentum_blend = 0.4 * momentum_3d + 0.35 * momentum_5d + 0.25 * momentum_7d
    
    # 20-day volatility using true range
    true_range = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    volatility_20d = true_range.rolling(window=20).mean()
    
    # 5-day volume acceleration (current vs 5-day average growth rate)
    volume_ma_5 = df['volume'].rolling(window=5).mean()
    volume_ma_10 = df['volume'].rolling(window=10).mean()
    volume_acceleration = (volume_ma_5 - volume_ma_10) / volume_ma_10
    
    # Volume trend (5-day vs 20-day ratio)
    volume_trend = volume_ma_5 / df['volume'].rolling(window=20).mean()
    
    # Clean interaction: momentum × volume trend ÷ volatility
    alpha_raw = momentum_blend * volume_trend / (volatility_20d + 1e-7)
    
    # Regime-aware smoothing: dynamic window based on volatility regime
    volatility_regime = volatility_20d / volatility_20d.rolling(window=50).mean()
    
    # Apply smoothing: more smoothing in high volatility, less in low volatility
    smooth_window = np.where(volatility_regime > 1.2, 5, 
                           np.where(volatility_regime < 0.8, 2, 3))
    
    # Vectorized smoothing with dynamic windows
    alpha_smoothed = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= max(smooth_window):
            window = int(smooth_window[i])
            alpha_smoothed.iloc[i] = alpha_raw.iloc[i-window+1:i+1].mean()
        else:
            alpha_smoothed.iloc[i] = alpha_raw.iloc[i]
    
    # Final factor with volume acceleration alignment
    alpha_factor = alpha_smoothed * (1 + volume_acceleration)
    
    return alpha_factor
