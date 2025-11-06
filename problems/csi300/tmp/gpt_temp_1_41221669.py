import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a composite alpha factor using hierarchical momentum, range efficiency, 
    volume-amount convergence, and price acceleration structures.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize final factor series
    alpha_factor = pd.Series(index=data.index, dtype=float)
    
    # 1. Hierarchical Momentum Structure
    # Primary Momentum Layer
    short_momentum = data['close'] / data['close'].shift(3) - 1
    medium_momentum = data['close'] / data['close'].shift(10) - 1
    long_momentum = data['close'] / data['close'].shift(20) - 1
    
    # Secondary Confirmation Layer
    volume_alignment = np.sign(short_momentum) * np.sign(data['volume'] / data['volume'].shift(1) - 1)
    amount_efficiency = short_momentum / ((data['amount'] / data['close']) - (data['amount'].shift(1) / data['close'].shift(1)))
    range_consistency = short_momentum * ((data['close'] - data['low']) / (data['high'] - data['low']))
    
    # Tertiary Persistence Layer
    momentum_acceleration = short_momentum - short_momentum.shift(1)
    
    # Volume persistence (count of volume increases in last 4 days)
    volume_persistence = pd.Series(0, index=data.index)
    for i in range(len(data)):
        if i >= 4:
            vol_increases = sum(data['volume'].iloc[i-j] > data['volume'].iloc[i-j-1] for j in range(4))
            volume_persistence.iloc[i] = vol_increases
    
    # Multi-timeframe convergence
    timeframe_convergence = (np.sign(short_momentum) * np.sign(medium_momentum) * np.sign(long_momentum))
    
    # Momentum hierarchy score
    momentum_hierarchy = (
        0.4 * short_momentum + 
        0.3 * medium_momentum + 
        0.3 * long_momentum +
        0.2 * volume_alignment +
        0.15 * amount_efficiency.fillna(0) +
        0.15 * range_consistency +
        0.1 * momentum_acceleration +
        0.05 * volume_persistence +
        0.05 * timeframe_convergence
    )
    
    # 2. Multi-Timeframe Range Efficiency
    # Intraday Efficiency Signals
    daily_range_util = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    close_position_strength = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    opening_gap_efficiency = abs(data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    
    # Multi-day Efficiency Patterns
    # 3-day cumulative efficiency
    cum_efficiency_3d = pd.Series(0.0, index=data.index)
    for i in range(len(data)):
        if i >= 2:
            price_change = abs(data['close'].iloc[i] - data['close'].iloc[i-3])
            range_sum = sum(data['high'].iloc[j] - data['low'].iloc[j] for j in range(i-2, i+1))
            cum_efficiency_3d.iloc[i] = price_change / range_sum if range_sum > 0 else 0
    
    # 5-day range expansion
    range_expansion = (data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5)).replace(0, np.nan)
    
    # Weekly close strength
    weekly_close_strength = pd.Series(0.0, index=data.index)
    for i in range(len(data)):
        if i >= 4:
            weekly_high = max(data['high'].iloc[i-4:i+1])
            weekly_close_strength.iloc[i] = data['close'].iloc[i] / weekly_high
    
    # Efficiency alignment score
    efficiency_alignment = (
        0.3 * daily_range_util.fillna(0) +
        0.25 * close_position_strength.fillna(0) +
        0.2 * opening_gap_efficiency.fillna(0) +
        0.15 * cum_efficiency_3d +
        0.1 * range_expansion.fillna(0)
    )
    
    # 3. Volume-Amount Convergence Hierarchy
    # Volume-Based Signals
    volume_trend_accel = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    volume_breakout = data['volume'] / data['volume'].rolling(window=4, min_periods=1).mean().shift(1)
    volume_persistence_bool = (data['volume'] > data['volume'].shift(1)) & (data['volume'].shift(1) > data['volume'].shift(2))
    
    # Amount-Based Signals
    amount_efficiency_signal = (data['close'] - data['close'].shift(1)) / data['amount'].replace(0, np.nan)
    amount_trend = data['amount'] / data['amount'].shift(1) - 1
    
    # Capital flow persistence
    capital_flow_persistence = pd.Series(0, index=data.index)
    for i in range(len(data)):
        if i >= 2:
            signs = [np.sign(data['amount'].iloc[j] / data['amount'].iloc[j-1] - 1) for j in range(i-1, i+1)]
            capital_flow_persistence.iloc[i] = 1 if len(set(signs)) == 1 else 0
    
    # Convergence Detection
    price_up = (data['close'] > data['close'].shift(1)).astype(int)
    volume_up = (data['volume'] > data['volume'].shift(1)).astype(int)
    amount_eff_positive = (amount_efficiency_signal > 0).astype(int)
    
    positive_convergence = price_up * volume_up * amount_eff_positive
    negative_convergence = ((data['close'] < data['close'].shift(1)).astype(int) * 
                          (data['volume'] < data['volume'].shift(1)).astype(int) * 
                          (amount_efficiency_signal < 0).astype(int))
    
    divergence_strength = amount_trend.fillna(0) * volume_trend_accel.fillna(0) * short_momentum
    
    # Convergence strength
    convergence_strength = (
        0.4 * positive_convergence +
        0.3 * (1 - negative_convergence) +
        0.3 * divergence_strength.fillna(0) +
        0.2 * volume_breakout.fillna(0) +
        0.15 * volume_persistence_bool.astype(float) +
        0.15 * capital_flow_persistence
    )
    
    # 4. Price Acceleration Structure
    # First-Order Momentum
    daily_return = data['close'] / data['close'].shift(1) - 1
    volume_weighted_return = daily_return * np.log(data['volume'].replace(0, 1))
    amount_efficiency_return = daily_return / (data['amount'] / data['close']).replace(0, np.nan)
    
    # Second-Order Acceleration
    momentum_change = daily_return - daily_return.shift(1)
    volume_confirmed_accel = momentum_change * (data['volume'] / data['volume'].shift(1)).replace(0, np.nan)
    amount_accel_alignment = momentum_change * np.sign(data['amount'] / data['amount'].shift(1) - 1)
    
    # Third-Order Persistence
    acceleration_consistency = momentum_change * momentum_change.shift(1)
    
    # Multi-day acceleration trend
    accel_trend_3d = momentum_change.rolling(window=3, min_periods=1).sum()
    
    # Volume-acceleration persistence
    vol_accel_persistence = pd.Series(0, index=data.index)
    for i in range(len(data)):
        if i >= 1:
            vol_trend = np.sign(data['volume'].iloc[i] / data['volume'].iloc[i-1] - 1)
            accel_sign = np.sign(momentum_change.iloc[i])
            vol_accel_persistence.iloc[i] = 1 if vol_trend == accel_sign else 0
    
    # Acceleration hierarchy score
    acceleration_hierarchy = (
        0.3 * daily_return +
        0.25 * volume_weighted_return.fillna(0) +
        0.2 * amount_efficiency_return.fillna(0) +
        0.15 * momentum_change.fillna(0) +
        0.1 * volume_confirmed_accel.fillna(0) +
        0.1 * vol_accel_persistence
    )
    
    # 5. Composite Signal Generation
    # Primary Strength Components
    primary_strength = (
        0.35 * momentum_hierarchy.fillna(0) +
        0.35 * efficiency_alignment.fillna(0) +
        0.3 * convergence_strength.fillna(0)
    )
    
    # Secondary Confirmation
    secondary_confirmation = (
        0.4 * acceleration_hierarchy.fillna(0) +
        0.3 * (close_position_strength.fillna(0) * range_expansion.fillna(0)) +
        0.3 * (amount_trend.fillna(0) * volume_persistence_bool.astype(float))
    )
    
    # Signal Quality Assessment
    # Consistency score (count aligned signals)
    consistency_score = pd.Series(0, index=data.index)
    for i in range(len(data)):
        if i >= 1:
            aligned_count = 0
            signals = [
                np.sign(momentum_hierarchy.iloc[i]),
                np.sign(efficiency_alignment.iloc[i]),
                np.sign(convergence_strength.iloc[i])
            ]
            if len(set(signals)) == 1:  # All same sign
                aligned_count = 3
            elif len(set(signals)) == 2:  # Two same sign
                aligned_count = 2
            consistency_score.iloc[i] = aligned_count
    
    # Final Alpha Output with hierarchical weighting
    high_conviction = (primary_strength > primary_strength.quantile(0.7)) & (secondary_confirmation > secondary_confirmation.quantile(0.6))
    medium_conviction = (primary_strength > primary_strength.quantile(0.5)) & (secondary_confirmation > secondary_confirmation.quantile(0.4))
    
    # Composite alpha factor
    alpha_factor = (
        (high_conviction.astype(float) * 1.5 * primary_strength) +
        (medium_conviction.astype(float) * primary_strength) +
        ((~high_conviction & ~medium_conviction).astype(float) * 0.7 * primary_strength) +
        0.2 * consistency_score * primary_strength
    )
    
    # Normalize the final factor
    alpha_factor = (alpha_factor - alpha_factor.mean()) / alpha_factor.std()
    
    return alpha_factor
