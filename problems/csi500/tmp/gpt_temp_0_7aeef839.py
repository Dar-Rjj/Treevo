import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Asymmetric Efficiency Analysis
    # Directional Volume Efficiency
    bull_volume_efficiency = np.where(
        data['close'] > data['open'],
        ((data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)) * data['volume'],
        0
    )
    
    bear_volume_efficiency = np.where(
        data['close'] < data['open'],
        ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * data['volume'],
        0
    )
    
    # Efficiency-Weighted Extremes
    close_position_ratio = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    efficiency_weighted_range = ((data['high'] - data['close']) + (data['close'] - data['low'])) / (data['high'] - data['low'] + 1e-8)
    
    # Microstructure Pressure Dynamics
    # Gap Absorption Pressure
    opening_gap_absorption = abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    closing_pressure = abs(data['close'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low'] + 1e-8)
    pressure_score = opening_gap_absorption + closing_pressure
    
    # Volume-Price Divergence
    volume_change_sign = np.sign(data['volume'] / data['volume'].shift(1) - 1)
    price_change_sign = np.sign(data['close'] / data['open'] - 1)
    volume_price_sign_alignment = volume_change_sign * price_change_sign
    
    # Range Momentum Convergence
    # Range Efficiency Momentum
    current_range_position = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    prev_range_position = (data['close'].shift(1) - data['low'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)
    range_position_momentum = current_range_position - prev_range_position
    
    # Range Acceleration Analysis
    current_range = data['high'] - data['low']
    prev_range = data['high'].shift(1) - data['low'].shift(1)
    prev_prev_range = data['high'].shift(2) - data['low'].shift(2)
    range_acceleration = (current_range / (prev_range + 1e-8)) - (prev_range / (prev_prev_range + 1e-8))
    
    # Volume Convergence Patterns
    # Multi-period Volume Dynamics
    short_term_volume_accel = data['volume'] / (data['volume'].shift(5) + 1e-8) - 1
    long_term_volume_accel = data['volume'] / (data['volume'].shift(10) + 1e-8) - 1
    volume_convergence = short_term_volume_accel - long_term_volume_accel
    
    # Component Integration
    # Core Efficiency Signal
    asymmetry_score = bull_volume_efficiency - bear_volume_efficiency
    efficiency_base = efficiency_weighted_range * (bull_volume_efficiency + bear_volume_efficiency)
    
    # Momentum-Enhanced Signal
    pressure_momentum_combined = pressure_score * range_position_momentum
    volume_convergence_weighted = pressure_momentum_combined * volume_convergence
    
    # Divergence Pattern Integration
    volume_price_alignment = volume_convergence_weighted * volume_price_sign_alignment
    range_acceleration_adjusted = volume_price_alignment * (1 + abs(range_acceleration))
    
    # Liquidity and Distribution Validation
    # Volume-Amount Efficiency
    volume_amount_ratio = data['volume'] / (data['amount'] + 1e-8)
    liquidity_filter = 1 / (1 + abs(volume_amount_ratio))
    
    # Since we don't have intraday data, we'll use rolling windows to approximate distribution patterns
    # Using 2-day rolling windows as approximation for morning/evening dominance
    morning_vs_evening_dominance = data['volume'].rolling(window=2).mean() / data['volume'].rolling(window=2, min_periods=1).apply(lambda x: x.iloc[-1] if len(x) > 0 else 1)
    
    # Volume acceleration pattern using 3-period rolling window
    volume_acceleration_pattern = (data['volume'].rolling(window=3).apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) == 3 else 0, raw=False) / 
                                  (data['volume'].rolling(window=3).apply(lambda x: x.iloc[0] if len(x) == 3 else 1, raw=False) + 1e-8))
    
    # Final Factor Construction
    # Pattern Persistence Validation
    volume_consistency_check = (np.sign(data['volume'] / data['volume'].shift(5) - 1) * 
                               np.sign(data['volume'].shift(1) / data['volume'].shift(6) - 1))
    
    distribution_persistence = morning_vs_evening_dominance * volume_acceleration_pattern
    
    # Composite Alpha Signal
    liquidity_refined_factor = range_acceleration_adjusted * liquidity_filter
    persistence_enhanced = liquidity_refined_factor * (1 + volume_consistency_check)
    final_output = -persistence_enhanced * distribution_persistence * close_position_ratio
    
    # Return as pandas Series with same index as input
    return pd.Series(final_output, index=data.index)
