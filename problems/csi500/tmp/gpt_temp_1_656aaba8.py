import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Aware Multi-Timeframe Momentum factor
    """
    # Calculate Price-Volume Interaction
    # Multiplicative Price-Volume Term
    price_volume = df['close'] * df['volume']
    
    # Apply Recursive Smoothing with Exponential Weighting
    alpha_ewm = 0.3  # Higher alpha prioritizes recent signals
    price_volume_smooth = price_volume.ewm(alpha=alpha_ewm).mean()
    
    # Blend with Volatility-Normalized Range
    # Calculate Daily Range
    daily_range = df['high'] - df['low']
    
    # Normalize by Volatility (20-day rolling volatility)
    volatility = df['close'].pct_change().rolling(window=20).std()
    normalized_range = daily_range / (volatility + 1e-8)
    
    # Apply Volume-Weighted Momentum
    # Calculate Momentum (5-day and 20-day)
    momentum_5d = df['close'].pct_change(periods=5)
    momentum_20d = df['close'].pct_change(periods=20)
    
    # Weight by Volume Confirmation using multi-timeframe volume
    volume_5d_avg = df['volume'].rolling(window=5).mean()
    volume_20d_avg = df['volume'].rolling(window=20).mean()
    
    volume_weighted_momentum = (
        momentum_5d * volume_5d_avg + 
        momentum_20d * volume_20d_avg
    )
    
    # Combine Components
    raw_factor = (
        0.4 * price_volume_smooth + 
        0.3 * normalized_range + 
        0.3 * volume_weighted_momentum
    )
    
    # Detect Market Regime
    # Volatility Regime (20-day rolling volatility percentile)
    vol_20d = df['close'].pct_change().rolling(window=20).std()
    vol_regime = vol_20d.rolling(window=60).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 70))
    )
    
    # Volume Regime (20-day rolling volume percentile)
    volume_20d_avg = df['volume'].rolling(window=20).mean()
    volume_regime = volume_20d_avg.rolling(window=60).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 70))
    )
    
    # Combine regimes (high volatility OR high volume = high regime)
    high_regime = (vol_regime == 1) | (volume_regime == 1)
    
    # Scale Factors by Regime
    # High Volatility/Volume: Reduce weight (0.7)
    # Low Volatility/Volume: Increase weight (1.3)
    regime_scaling = np.where(high_regime, 0.7, 1.3)
    
    # Apply regime scaling
    final_factor = raw_factor * regime_scaling
    
    return final_factor
