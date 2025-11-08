import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate Multi-Timeframe Intraday Price Acceleration
    # Intraday Price Velocity (Short-term)
    intraday_velocity = (data['close'] - data['open']) / (data['high'] - data['low'])
    intraday_velocity = intraday_velocity.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Intraday Price Velocity (Medium-term) - 5-day average
    velocity_5d = intraday_velocity.rolling(window=5, min_periods=1).mean()
    
    # Intraday Acceleration
    short_term_acceleration = intraday_velocity - intraday_velocity.shift(1)
    medium_term_acceleration = velocity_5d - velocity_5d.shift(1)
    
    # Calculate Multi-Timeframe Volume Momentum
    volume_5d_ma = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_20d_ma = data['volume'].rolling(window=20, min_periods=1).mean()
    
    short_term_volume_momentum = data['volume'] / volume_5d_ma
    medium_term_volume_momentum = data['volume'] / volume_20d_ma
    
    # Volatility Regime Identification
    # Calculate 5-day volatility using High, Low, Close prices
    daily_range = (data['high'] - data['low']) / data['close']
    volatility_5d = daily_range.rolling(window=5, min_periods=1).std()
    
    # Classify regime based on volatility percentiles
    volatility_percentile = volatility_5d.rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    high_vol_regime = volatility_percentile > 0.7
    low_vol_regime = volatility_percentile < 0.3
    
    # Detect Multi-Timeframe Acceleration-Volume Divergence
    # Short-term divergence detection
    short_term_divergence = (
        (short_term_acceleration > 0) & (short_term_volume_momentum < 1) |
        (short_term_acceleration < 0) & (short_term_volume_momentum > 1)
    )
    
    # Medium-term divergence detection
    medium_term_divergence = (
        (medium_term_acceleration > 0) & (medium_term_volume_momentum.diff() < 0) |
        (medium_term_acceleration < 0) & (medium_term_volume_momentum.diff() > 0)
    )
    
    # Construct Regime-Adaptive Composite Factor
    # High volatility regime factor - mean reversion emphasis
    high_vol_factor = -1 * (short_term_acceleration * short_term_volume_momentum)
    
    # Low volatility regime factor - trend continuation emphasis
    low_vol_factor = medium_term_acceleration * medium_term_volume_momentum
    
    # Combine factors based on volatility regime
    regime_factor = pd.Series(index=data.index, dtype=float)
    regime_factor[high_vol_regime] = high_vol_factor[high_vol_regime]
    regime_factor[low_vol_regime] = low_vol_factor[low_vol_regime]
    regime_factor[~high_vol_regime & ~low_vol_regime] = 0.5 * high_vol_factor[~high_vol_regime & ~low_vol_regime] + 0.5 * low_vol_factor[~high_vol_regime & ~low_vol_regime]
    
    # Scale by absolute intraday velocity for magnitude adjustment
    final_factor = regime_factor * np.abs(intraday_velocity)
    
    # Generate Predictive Signal
    # Apply divergence conditions to enhance signal quality
    signal_enhancement = pd.Series(index=data.index, dtype=float)
    signal_enhancement[high_vol_regime & short_term_divergence] = 1.5
    signal_enhancement[low_vol_regime & ~medium_term_divergence] = 1.2
    signal_enhancement[~high_vol_regime & ~low_vol_regime] = 1.0
    
    enhanced_factor = final_factor * signal_enhancement
    
    return enhanced_factor.fillna(0)
