import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Extract price and volume data
    close = df['close']
    volume = df['volume']
    
    # Multi-Timeframe Momentum Signals
    mom_3d = close / close.shift(3) - 1
    mom_8d = close / close.shift(8) - 1
    mom_15d = close / close.shift(15) - 1
    
    # Calculate returns for volatility computation
    returns = close.pct_change()
    
    # Volatility-Adjusted Regime Detection
    recent_vol = returns.shift(1).rolling(window=5).std()
    baseline_vol = returns.shift(1).rolling(window=20).std()
    
    # Volatility Regime
    high_vol_regime = recent_vol > baseline_vol
    low_vol_regime = ~high_vol_regime
    
    # Nonlinear Volume-Price Divergence
    # Volume Momentum Components
    fast_volume = volume / volume.shift(3) - 1
    slow_volume = volume / volume.shift(15) - 1
    
    # Volume Divergence Signal
    volume_divergence = fast_volume - slow_volume
    
    # Price-Volume Relationship Strength
    # Calculate daily returns and volume changes
    daily_returns = close.pct_change()
    daily_volume_changes = volume.pct_change()
    
    # 8-day Rolling Correlation
    correlation = pd.Series(index=df.index, dtype=float)
    for i in range(8, len(df)):
        window_returns = daily_returns.iloc[i-7:i+1]
        window_volume = daily_volume_changes.iloc[i-7:i+1]
        valid_mask = (~window_returns.isna()) & (~window_volume.isna())
        if valid_mask.sum() >= 5:  # Minimum 5 valid observations
            correlation.iloc[i] = window_returns[valid_mask].corr(window_volume[valid_mask])
        else:
            correlation.iloc[i] = 0
    
    # Correlation Regime
    strong_correlation = correlation.abs() > 0.3
    weak_correlation = ~strong_correlation
    
    # Adaptive Alpha Construction
    # Regime-Based Momentum Selection
    selected_momentum = pd.Series(index=df.index, dtype=float)
    selected_momentum[high_vol_regime] = mom_3d[high_vol_regime]
    selected_momentum[low_vol_regime] = mom_15d[low_vol_regime]
    
    # Volume-Weighted Enhancement
    volume_weighted_momentum = selected_momentum * volume_divergence
    
    # Correlation-Based Confidence Adjustment
    factor = pd.Series(index=df.index, dtype=float)
    factor[strong_correlation] = volume_weighted_momentum[strong_correlation]
    factor[weak_correlation] = volume_weighted_momentum[weak_correlation] * 0.5
    
    # Handle NaN values
    factor = factor.fillna(0)
    
    return factor
