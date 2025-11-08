import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Horizon Momentum Calculation
    short_momentum = df['close'] / df['close'].shift(5) - 1
    medium_momentum = df['close'] / df['close'].shift(15) - 1
    long_momentum = df['close'] / df['close'].shift(30) - 1
    
    # Robust Volatility Scaling
    daily_range_vol = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Rolling Volatility Estimation using Median Absolute Deviation
    def mad_volatility(series):
        return (series - series.median()).abs().median() * 1.4826
    
    rolling_vol = daily_range_vol.rolling(window=10, min_periods=5).apply(
        mad_volatility, raw=False
    )
    
    # Volatility-Normalized Momentum
    short_norm = short_momentum / rolling_vol
    medium_norm = medium_momentum / rolling_vol
    long_norm = long_momentum / rolling_vol
    
    # Volume Regime Confirmation
    volume_momentum = df['volume'] / df['volume'].shift(5) - 1
    volume_acceleration = volume_momentum - volume_momentum.shift(5)
    
    # Nonlinear Volume Transform
    volume_transform = np.tanh(volume_momentum) * np.sign(volume_acceleration)
    
    # Adaptive Combination Framework
    # Momentum Fusion using geometric mean with cube root for stability
    fused_momentum = np.cbrt(short_norm * medium_norm * long_norm)
    
    # Volume Confirmation Weighting (map to [0.3, 1.0])
    volume_weight = 0.35 * volume_transform + 0.65
    
    # Final Alpha Factor
    alpha_factor = fused_momentum * volume_weight
    
    return alpha_factor
