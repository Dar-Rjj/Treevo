import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Extract price and volume data
    close = df['close']
    volume = df['volume']
    
    # Initialize output series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate multi-timeframe returns
    R5 = close.pct_change(5)
    R10 = close.pct_change(10)
    R20 = close.pct_change(20)
    
    # Momentum Quality Assessment
    momentum_consistency = ((R5 > 0) & (R10 > 0) & (R20 > 0)) | ((R5 < 0) & (R10 < 0) & (R20 < 0))
    momentum_strength = (abs(R5) + abs(R10) + abs(R20)) / 3
    
    # Adaptive Volatility Estimation
    returns_1d = close.pct_change()
    short_vol = returns_1d.rolling(window=10).std()
    medium_vol = returns_1d.rolling(window=20).std()
    volatility_ratio = short_vol / medium_vol
    
    # Regime Classification
    high_vol_regime = volatility_ratio > 1.2
    low_vol_regime = volatility_ratio < 0.8
    normal_vol_regime = (volatility_ratio >= 0.8) & (volatility_ratio <= 1.2)
    
    # Volume Acceleration Profile
    sma_volume_5 = volume.rolling(window=5).mean()
    sma_volume_20 = volume.rolling(window=20).mean()
    V5 = volume / sma_volume_5.shift(1) - 1
    V20 = volume / sma_volume_20.shift(1) - 1
    volume_acceleration = V5 - V20
    
    # Volume-Price Alignment
    volume_price_corr = pd.Series(index=df.index, dtype=float)
    for i in range(20, len(df)):
        if i >= 30:  # Need at least 10 data points for correlation
            returns_window = returns_1d.iloc[i-10:i]
            volume_window = volume.iloc[i-10:i]
            if len(returns_window) >= 5 and len(volume_window) >= 5:  # Minimum data requirement
                corr = returns_window.corr(volume_window)
                volume_price_corr.iloc[i] = corr if not np.isnan(corr) else 0
            else:
                volume_price_corr.iloc[i] = 0
        else:
            volume_price_corr.iloc[i] = 0
    
    alignment_score = np.sign(R5) * np.sign(volume_acceleration) * abs(volume_price_corr)
    
    # Volume Regime Detection
    high_volume_regime = (V5 > 0.1) & (volume_acceleration > 0)
    low_volume_regime = (V5 < -0.1) & (volume_acceleration < 0)
    neutral_volume_regime = ~(high_volume_regime | low_volume_regime)
    
    # Regime-Weighted Momentum
    regime_weighted_momentum = pd.Series(index=df.index, dtype=float)
    regime_weighted_momentum[high_vol_regime] = 0.7 * R5[high_vol_regime] + 0.3 * R10[high_vol_regime]
    regime_weighted_momentum[low_vol_regime] = 0.3 * R5[low_vol_regime] + 0.7 * R20[low_vol_regime]
    regime_weighted_momentum[normal_vol_regime] = 0.4 * R5[normal_vol_regime] + 0.4 * R10[normal_vol_regime] + 0.2 * R20[normal_vol_regime]
    
    # Volume Confirmation Multiplier
    volume_multiplier = pd.Series(1.0, index=df.index)
    strong_confirmation = alignment_score > 0.5
    weak_confirmation = (alignment_score >= -0.5) & (alignment_score <= 0.5)
    contradiction = alignment_score < -0.5
    
    volume_multiplier[strong_confirmation] = 1 + volume_acceleration[strong_confirmation]
    volume_multiplier[contradiction] = 1 - volume_acceleration[contradiction]
    
    # Final Alpha Factor
    base_alpha = regime_weighted_momentum * volume_multiplier
    final_alpha = base_alpha * momentum_consistency.astype(float) * momentum_strength
    
    return final_alpha
