import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Regime-Adaptive Momentum with Volume Acceleration
    # Combines volatility-normalized momentum with volume acceleration using adaptive windows
    # Interpretable as: Stocks with strong momentum signals amplified by accelerating volume in appropriate volatility regimes
    
    # Adaptive window selection based on recent volatility regime
    returns = df['close'].pct_change()
    short_vol = returns.rolling(window=5).std()
    long_vol = returns.rolling(window=20).std()
    vol_regime = short_vol / (long_vol + 1e-7)
    
    # Dynamic momentum windows: shorter in high vol, longer in low vol
    momentum_window = np.where(vol_regime > 1.2, 2, 
                              np.where(vol_regime < 0.8, 5, 3))
    
    # Volatility-normalized momentum with adaptive windows
    price_momentum = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= max(momentum_window):
            window = int(momentum_window[i])
            mom = (df['close'].iloc[i] - df['close'].iloc[i-window]) / df['close'].iloc[i-window]
            vol = returns.iloc[i-window:i+1].std()
            price_momentum.iloc[i] = mom / (vol + 1e-7)
    
    # Volume acceleration with multiplicative weighting
    volume_3d = df['volume'].rolling(window=3).mean()
    volume_10d = df['volume'].rolling(window=10).mean()
    volume_acceleration = (volume_3d - volume_10d) / (volume_10d + 1e-7)
    
    # Volume-weighted momentum using regime-adaptive volume multiplier
    volume_multiplier = 1 + np.abs(volume_acceleration) * np.sign(volume_acceleration)
    
    # Range-based momentum confirmation
    daily_ranges = df['high'] - df['low']
    range_momentum = (daily_ranges - daily_ranges.shift(3)) / (daily_ranges.shift(3) + 1e-7)
    
    # Final alpha: volatility-normalized momentum amplified by volume acceleration
    alpha = price_momentum * volume_multiplier + 0.3 * range_momentum
    
    return alpha
