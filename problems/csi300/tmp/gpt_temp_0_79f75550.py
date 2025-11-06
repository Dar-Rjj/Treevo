import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Structure
    close = df['close']
    
    # Multi-Timeframe Returns
    very_short = close.pct_change(periods=2)
    short_term = close.pct_change(periods=5)
    medium_term = close.pct_change(periods=10)
    
    # Momentum Acceleration
    primary_acceleration = very_short - short_term
    secondary_acceleration = short_term - medium_term
    momentum_curvature = very_short - 2 * short_term + medium_term
    
    # Momentum Quality
    direction_consistency = pd.Series(
        [sum((very_short.shift(i) * very_short.shift(i+1)) > 0 for i in range(3)) 
         for j in range(len(close))], index=close.index
    )
    acceleration_consistency = pd.Series(
        [sum((primary_acceleration.shift(i) * primary_acceleration.shift(i+1)) > 0 for i in range(3)) 
         for j in range(len(close))], index=close.index
    )
    momentum_strength = very_short / medium_term.replace(0, np.nan)
    
    # Volume Confirmation Framework
    volume = df['volume']
    volume_change = volume.pct_change()
    volume_trend = volume.pct_change(periods=3)
    volume_acceleration = volume_change - volume_change.shift(1)
    
    # Price-Volume Relationship
    volume_confirmation = ((very_short > 0) & (volume_change > 0)) | ((very_short < 0) & (volume_change < 0))
    divergence = ((very_short > 0) & (volume_change < 0)) | ((very_short < 0) & (volume_change > 0))
    divergence_magnitude = abs(very_short) * abs(volume_change)
    
    # Volume Regime
    high_volume = volume_change > 0.5
    low_volume = volume_change < -0.3
    volume_persistence = pd.Series(
        [sum(volume_change.shift(i) > 0 for i in range(3)) 
         for j in range(len(close))], index=close.index
    )
    
    # Volatility Regime System
    daily_range = (df['high'] - df['low']) / close
    short_volatility = daily_range.rolling(window=3).std()
    volatility_change = short_volatility.pct_change()
    
    # Volatility State
    high_volatility = short_volatility > 0.02
    low_volatility = short_volatility < 0.005
    volatility_persistence = pd.Series(
        [sum((short_volatility.shift(i) > 0.02) == (short_volatility.shift(i+1) > 0.02) for i in range(3)) 
         for j in range(len(close))], index=close.index
    )
    
    # Volatility-Adjusted Signals
    risk_adjusted_momentum = very_short / short_volatility.replace(0, np.nan)
    volatility_breakout = volatility_change > 0.3
    stable_regime = low_volatility.astype(int) * volatility_persistence
    
    # Cross-Factor Interactions
    # Momentum-Volume Interactions
    confirmed_momentum = very_short * volume_confirmation.astype(int)
    acceleration_with_volume = primary_acceleration * volume_change
    divergence_warning = momentum_curvature * divergence_magnitude
    
    # Momentum-Volatility Interactions
    stable_momentum = very_short * stable_regime
    volatility_breakout_momentum = primary_acceleration * volatility_breakout.astype(int)
    risk_adjusted_acceleration = primary_acceleration / short_volatility.replace(0, np.nan)
    
    # Volume-Volatility Interactions
    high_volume_volatility = high_volume.astype(int) * volatility_breakout.astype(int)
    low_volume_stability = low_volume.astype(int) * low_volatility.astype(int)
    volume_volatility_alignment = volume_persistence * volatility_persistence
    
    # Three-Way Regime Alignment
    optimal_conditions = direction_consistency * volume_confirmation.astype(int) * stable_regime
    breakout_conditions = acceleration_consistency * high_volume.astype(int) * volatility_breakout.astype(int)
    warning_conditions = divergence.astype(int) * high_volatility.astype(int) * low_volume.astype(int)
    
    # Composite Alpha Construction
    # Core Components
    base_signal = risk_adjusted_momentum
    volume_adjusted = base_signal * (1 + volume_confirmation.astype(int))
    regime_scaling = volume_adjusted * optimal_conditions
    
    # Enhancement Layer
    composite = regime_scaling
    composite = composite + primary_acceleration  # Acceleration Overlay
    composite = composite * (1 + breakout_conditions * 0.5)  # Breakout Boost
    composite = composite * (1 - warning_conditions * 0.3)  # Warning Filter
    
    return composite
