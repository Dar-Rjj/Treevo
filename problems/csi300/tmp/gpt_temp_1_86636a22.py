import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining regime-adaptive momentum with volume confirmation,
    persistent low-volatility acceleration, and volume-stabilized breakout signals.
    """
    # Extract price and volume data
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Calculate basic returns and ranges
    returns_1d = close.pct_change()
    returns_3d = close.pct_change(3)
    returns_5d = close.pct_change(5)
    daily_range = (high - low) / close
    
    # Volume calculations
    volume_change_1d = volume.pct_change()
    volume_change_5d = volume.pct_change(5)
    volume_ma_5d = volume.rolling(window=5).mean()
    
    # Momentum components
    acceleration = returns_1d - returns_5d
    momentum_persistence = returns_1d.rolling(window=3).apply(
        lambda x: sum(np.sign(x) == np.sign(x.shift(1))), raw=False
    )
    trend_identification = (np.sign(returns_5d) == np.sign(returns_1d)).astype(int)
    
    # Volume components
    volume_confirmation = (np.sign(returns_1d) == np.sign(volume_change_1d)).astype(int)
    volume_consistency = volume.rolling(window=3).apply(
        lambda x: sum(x > x.shift(1)), raw=False
    )
    volume_regime = (volume > volume_ma_5d.shift(1)).astype(int)
    
    # Volatility components
    volatility_regime = (daily_range > daily_range.rolling(window=5).mean().shift(1)).astype(int)
    volatility_stability = daily_range / (daily_range.rolling(window=5).std() + 0.001)
    
    # Breakout detection
    breakout_detection = (high > high.rolling(window=5).max().shift(1)).astype(int)
    
    # Multi-timeframe integration
    # 1. Regime-adaptive momentum with volume confirmation
    regime_adaptive_momentum = returns_1d * (1 - volatility_regime) * volume_confirmation
    
    # 2. Persistent low-volatility acceleration
    persistent_lowvol_acceleration = acceleration * momentum_persistence * (1 - volatility_regime)
    
    # 3. Volume-stabilized breakout signals
    volume_stabilized_breakout = breakout_detection * volume_regime * volatility_stability
    
    # 4. Multi-timeframe confirmed momentum
    multi_timeframe_momentum = (returns_1d * returns_3d) * volume_confirmation * trend_identification
    
    # Combine factors with weights based on economic intuition
    alpha_factor = (
        0.4 * regime_adaptive_momentum +
        0.3 * persistent_lowvol_acceleration +
        0.2 * volume_stabilized_breakout +
        0.1 * multi_timeframe_momentum
    )
    
    # Normalize the factor
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=20).mean()) / alpha_factor.rolling(window=20).std()
    
    return alpha_factor
