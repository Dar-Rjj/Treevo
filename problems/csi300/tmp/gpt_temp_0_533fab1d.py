import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Momentum Calculation
    close = df['close']
    short_momentum = (close / close.shift(3)) - 1
    medium_momentum = (close / close.shift(10)) - 1
    long_momentum = (close / close.shift(20)) - 1
    
    # Adaptive Volatility Scaling
    range_vol = (df['high'] - df['low']) / df['close']
    avg_range_vol_5d = range_vol.rolling(window=5).mean()
    avg_range_vol_20d = range_vol.rolling(window=20).mean()
    return_vol_10d = close.pct_change().rolling(window=10).std()
    
    # Volatility regime detection using rolling percentiles
    vol_percentile = avg_range_vol_5d.rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Regime classification
    high_vol_regime = vol_percentile > 0.7
    low_vol_regime = vol_percentile < 0.3
    medium_vol_regime = ~high_vol_regime & ~low_vol_regime
    
    # Regime-specific scaling
    short_scaled = short_momentum.copy()
    medium_scaled = medium_momentum.copy()
    long_scaled = long_momentum.copy()
    
    short_scaled[high_vol_regime] = short_momentum[high_vol_regime] / avg_range_vol_5d[high_vol_regime]
    short_scaled[medium_vol_regime] = short_momentum[medium_vol_regime] / return_vol_10d[medium_vol_regime]
    short_scaled[low_vol_regime] = short_momentum[low_vol_regime] / avg_range_vol_20d[low_vol_regime]
    
    medium_scaled[high_vol_regime] = medium_momentum[high_vol_regime] / avg_range_vol_5d[high_vol_regime]
    medium_scaled[medium_vol_regime] = medium_momentum[medium_vol_regime] / return_vol_10d[medium_vol_regime]
    medium_scaled[low_vol_regime] = medium_momentum[low_vol_regime] / avg_range_vol_20d[low_vol_regime]
    
    long_scaled[high_vol_regime] = long_momentum[high_vol_regime] / avg_range_vol_5d[high_vol_regime]
    long_scaled[medium_vol_regime] = long_momentum[medium_vol_regime] / return_vol_10d[medium_vol_regime]
    long_scaled[low_vol_regime] = long_momentum[low_vol_regime] / avg_range_vol_20d[low_vol_regime]
    
    # Volume-Price Alignment Analysis
    volume = df['volume']
    short_vol_momentum = volume / volume.shift(3)
    medium_vol_trend = volume / volume.shift(10)
    vol_acceleration = (volume / volume.shift(3)) / (volume.shift(3) / volume.shift(6))
    
    # Price-volume alignment signals
    strong_alignment = (short_momentum > 0) & (short_vol_momentum > 1.1)
    weak_alignment = (short_momentum > 0) & (short_vol_momentum > 0.9)
    divergence = (short_momentum > 0) & (short_vol_momentum < 0.8)
    negative_alignment = (short_momentum < 0) & (short_vol_momentum > 1.1)
    
    # Regime-Adaptive Momentum Combination
    regime_momentum = pd.Series(index=df.index, dtype=float)
    
    # High volatility regime weights
    high_vol_momentum = (0.7 * short_scaled[high_vol_regime] + 
                         0.2 * medium_scaled[high_vol_regime] + 
                         0.1 * long_scaled[high_vol_regime])
    
    # Medium volatility regime weights
    medium_vol_momentum = (0.4 * short_scaled[medium_vol_regime] + 
                           0.4 * medium_scaled[medium_vol_regime] + 
                           0.2 * long_scaled[medium_vol_regime])
    
    # Low volatility regime weights
    low_vol_momentum = (0.2 * short_scaled[low_vol_regime] + 
                        0.3 * medium_scaled[low_vol_regime] + 
                        0.5 * long_scaled[low_vol_regime])
    
    regime_momentum[high_vol_regime] = high_vol_momentum
    regime_momentum[medium_vol_regime] = medium_vol_momentum
    regime_momentum[low_vol_regime] = low_vol_momentum
    
    # Volume Confirmation Application
    alignment_multiplier = pd.Series(1.0, index=df.index)
    alignment_multiplier[strong_alignment] = 1.4
    alignment_multiplier[weak_alignment] = 1.1
    alignment_multiplier[divergence] = 0.6
    alignment_multiplier[negative_alignment] = 0.3
    
    # Volume stability adjustment
    volume_5d_std = volume.rolling(window=5).std()
    volume_5d_mean = volume.rolling(window=5).mean()
    volume_variability = volume_5d_std / volume_5d_mean
    stability_multiplier = 1 / (1 + volume_variability)
    
    # Final Alpha Construction
    alpha = (regime_momentum * alignment_multiplier * stability_multiplier * 
             np.log(df['amount']))
    
    return alpha
