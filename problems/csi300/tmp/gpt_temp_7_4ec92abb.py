import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining price momentum acceleration, volume confirmation,
    forward return alignment, and volatility adjustment.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Price Momentum Acceleration System
    # Ultra-short Acceleration (2-day)
    current_momentum_ultra = (close - close.shift(2)) / close.shift(2)
    prev_momentum_ultra = (close.shift(2) - close.shift(4)) / close.shift(4)
    ultra_short_accel = current_momentum_ultra - prev_momentum_ultra
    
    # Short-term Acceleration (5-day)
    current_momentum_short = (close - close.shift(5)) / close.shift(5)
    prev_momentum_short = (close.shift(5) - close.shift(10)) / close.shift(10)
    short_term_accel = current_momentum_short - prev_momentum_short
    
    # Acceleration Alignment
    acceleration_alignment = ultra_short_accel * short_term_accel
    
    # Price Range Breakout Detection
    recent_range = (high - low).rolling(window=10).mean()
    range_threshold = recent_range * 0.6
    
    upper_breakout = (high - high.rolling(window=10).max().shift(1)) / range_threshold
    lower_breakdown = (low.rolling(window=10).min().shift(1) - low) / range_threshold
    net_breakout = upper_breakout - lower_breakdown
    
    # Price Signal
    price_signal = acceleration_alignment * net_breakout
    
    # Volume-Price Confirmation System
    # Volume Acceleration Framework
    short_term_vol_momentum = volume / volume.rolling(window=5).mean() - 1
    medium_term_vol_trend = volume / volume.rolling(window=15).mean() - 1
    volume_acceleration = short_term_vol_momentum * medium_term_vol_trend
    
    # Volume-Price Divergence
    price_direction = np.sign(close - close.shift(1))
    volume_direction = np.sign(volume - volume.shift(1))
    direction_alignment = price_direction * volume_direction
    volume_quality = volume_acceleration * direction_alignment
    
    # Forward Return Alignment Engine
    # Multi-horizon Forward Returns
    forward_1d = (close.shift(-1) - close) / close
    forward_3d = (close.shift(-3) - close) / close
    forward_5d = (close.shift(-5) - close) / close
    
    # Dynamic Horizon Selection
    corr_1d = price_signal.rolling(window=15).corr(forward_1d)
    corr_3d = price_signal.rolling(window=15).corr(forward_3d)
    corr_5d = price_signal.rolling(window=15).corr(forward_5d)
    
    # Optimal Horizon selection
    corr_df = pd.DataFrame({
        'corr_1d': corr_1d.abs(),
        'corr_3d': corr_3d.abs(),
        'corr_5d': corr_5d.abs()
    })
    
    optimal_horizon = corr_df.idxmax(axis=1)
    forward_return = pd.Series(index=close.index, dtype=float)
    
    for idx in forward_return.index:
        if optimal_horizon.loc[idx] == 'corr_1d':
            forward_return.loc[idx] = forward_1d.loc[idx]
        elif optimal_horizon.loc[idx] == 'corr_3d':
            forward_return.loc[idx] = forward_3d.loc[idx]
        else:
            forward_return.loc[idx] = forward_5d.loc[idx]
    
    # Adaptive Signal Enhancement
    signal_performance = price_signal * forward_return.shift(1)  # Use lagged forward return
    recent_performance = signal_performance.rolling(window=10).mean()
    
    performance_direction = np.sign(recent_performance)
    performance_strength = np.minimum(np.abs(recent_performance) * 10, 2)
    adaptive_weight = 1 + performance_direction * performance_strength
    
    # Volatility-Adjusted Scaling
    short_term_vol = close.rolling(window=10).std()
    medium_term_vol = close.rolling(window=20).std()
    volatility_ratio = short_term_vol / medium_term_vol
    
    base_adjustment = 1 / (short_term_vol + 0.0001)
    regime_adjustment = base_adjustment * np.where(volatility_ratio > 1, 1.2, 0.8)
    
    # Final Alpha Construction
    core_signal = price_signal * volume_quality
    forward_aligned = core_signal * forward_return
    volatility_scaled = forward_aligned * regime_adjustment
    alpha_factor = volatility_scaled * adaptive_weight
    
    return alpha_factor
