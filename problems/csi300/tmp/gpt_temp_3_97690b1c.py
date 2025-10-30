import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum-Efficiency Divergence with Volume-Pressure Confirmation
    """
    data = df.copy()
    
    # Multi-Period Momentum Divergence Analysis
    # Calculate directional momentum across asymmetric windows
    momentum_5 = (data['close'] / data['close'].shift(5) - 1) * 100
    momentum_13 = (data['close'] / data['close'].shift(13) - 1) * 100
    momentum_34 = (data['close'] / data['close'].shift(34) - 1) * 100
    
    # Momentum divergence patterns
    momentum_divergence = (
        (momentum_5 - momentum_13.rolling(5).mean()) + 
        (momentum_13 - momentum_34.rolling(13).mean()) +
        (momentum_5.rolling(3).mean() - momentum_5.rolling(8).mean())
    )
    
    # Momentum acceleration/deceleration
    momentum_accel = (
        momentum_5 - momentum_5.shift(3) + 
        momentum_13 - momentum_13.shift(5) +
        momentum_34 - momentum_34.shift(8)
    )
    
    # Efficiency-Pressure Divergence Detection
    # Pressure efficiency: close-to-close momentum vs range
    daily_range = (data['high'] - data['low']) / data['close']
    pressure_efficiency_5 = momentum_5 / (daily_range.rolling(5).mean() + 1e-8)
    pressure_efficiency_13 = momentum_13 / (daily_range.rolling(13).mean() + 1e-8)
    
    # Buying/selling pressure intensity
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    volume_pressure = data['volume'] * (typical_price - typical_price.shift(1))
    buying_pressure = volume_pressure.rolling(5).apply(lambda x: x[x > 0].sum() if len(x[x > 0]) > 0 else 0)
    selling_pressure = volume_pressure.rolling(5).apply(lambda x: abs(x[x < 0].sum()) if len(x[x < 0]) > 0 else 0)
    pressure_ratio = buying_pressure / (selling_pressure + 1e-8)
    
    # Efficiency-pressure divergence
    efficiency_divergence = (
        pressure_efficiency_5 - pressure_efficiency_13 + 
        pressure_efficiency_5.rolling(3).mean() - pressure_efficiency_5.rolling(8).mean()
    )
    
    # Volume-Volatility Confirmation Framework
    # Volume patterns relative to recent volatility
    volume_ma_5 = data['volume'].rolling(5).mean()
    volume_ma_20 = data['volume'].rolling(20).mean()
    volume_intensity = volume_ma_5 / (volume_ma_20 + 1e-8)
    
    # Volatility regimes
    volatility_5 = data['close'].rolling(5).std() / data['close'].rolling(5).mean()
    volatility_20 = data['close'].rolling(20).std() / data['close'].rolling(20).mean()
    vol_regime = volatility_5 / (volatility_20 + 1e-8)
    
    # Volume breakouts with volatility context
    volume_breakout = (data['volume'] > volume_ma_20 * 1.2) & (vol_regime > 0.8)
    volume_suppression = (data['volume'] < volume_ma_20 * 0.8) & (vol_regime < 1.2)
    
    # Volume-volatility efficiency matching
    vol_vol_efficiency = volume_intensity / (vol_regime + 1e-8)
    
    # Adaptive Alpha Synthesis
    # Combine momentum divergence with efficiency-pressure signals
    momentum_efficiency_composite = (
        momentum_divergence * 0.4 + 
        momentum_accel * 0.3 + 
        efficiency_divergence * 0.3
    )
    
    # Volume-volatility confirmation as weighting mechanism
    volume_confirmation_weight = (
        np.where(volume_breakout, 1.5, 1.0) * 
        np.where(volume_suppression, 0.7, 1.0) * 
        np.tanh(vol_vol_efficiency)
    )
    
    # Cross-timeframe signal consistency scoring
    timeframe_consistency = (
        (np.sign(momentum_5) == np.sign(momentum_13)).astype(int) * 0.3 +
        (np.sign(momentum_13) == np.sign(momentum_34)).astype(int) * 0.3 +
        (np.sign(pressure_efficiency_5) == np.sign(pressure_efficiency_13)).astype(int) * 0.4
    )
    
    # Final alpha generation with regime detection
    alpha = (
        momentum_efficiency_composite * 
        volume_confirmation_weight * 
        timeframe_consistency * 
        pressure_ratio.rolling(3).mean()
    )
    
    # Normalize and smooth the final alpha
    alpha_normalized = (alpha - alpha.rolling(50).mean()) / (alpha.rolling(50).std() + 1e-8)
    alpha_smoothed = alpha_normalized.rolling(3).mean()
    
    return alpha_smoothed
