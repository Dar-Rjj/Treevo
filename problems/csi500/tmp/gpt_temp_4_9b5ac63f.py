import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volume-Confirmed Trend Persistence factor
    """
    data = df.copy()
    
    # Trend Persistence Analysis
    # Short-term Trend Strength (3-day)
    data['price_change'] = data['close'].diff()
    data['direction'] = np.sign(data['price_change'])
    data['consecutive_direction'] = data['direction'].groupby(data.index).transform(
        lambda x: (x == x.shift(1)).cumsum() * (x == x.shift(1))
    )
    data['net_directional_days'] = data['close'].rolling(window=3).apply(
        lambda x: (x.diff().dropna() > 0).sum() - (x.diff().dropna() < 0).sum()
    )
    
    # Medium-term Trend Consistency (10-day)
    data['directional_consistency'] = data['close'].rolling(window=10).apply(
        lambda x: (x > x.iloc[0]).sum() - (x < x.iloc[0]).sum()
    )
    data['price_momentum_persistence'] = (
        np.sign(data['close'] / data['close'].shift(10) - 1) * 
        abs(data['close'] / data['close'].shift(10) - 1)
    )
    
    # Trend Persistence Score
    data['trend_persistence_score'] = (
        (data['consecutive_direction'].fillna(0) + data['net_directional_days'].fillna(0)) *
        (data['directional_consistency'].fillna(0) + data['price_momentum_persistence'].fillna(0))
    )
    
    # Volume Confirmation Analysis
    data['volume_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['volume_trend'] = data['volume'] / data['volume_5d_avg']
    data['volume_above_avg'] = data['volume'] > data['volume_5d_avg']
    data['volume_persistence'] = data['volume_above_avg'].groupby(data.index).transform(
        lambda x: (x == x.shift(1)).cumsum() * (x == x.shift(1))
    )
    
    # Volume-Price Efficiency
    data['daily_range_efficiency'] = (
        (data['close'] - data['close'].shift(1)) / 
        (data['high'] - data['low']).replace(0, np.nan)
    )
    
    volume_10d_avg = data['volume'].rolling(window=10).mean()
    volume_10d_std = data['volume'].rolling(window=10).std()
    data['volume_spike_quality'] = data['volume'] / (volume_10d_avg + 2 * volume_10d_std)
    
    # Volume Confirmation Score
    data['volume_confirmation_score'] = (
        data['volume_trend'].fillna(1) * 
        data['volume_persistence'].fillna(0) *
        data['daily_range_efficiency'].fillna(0) *
        data['volume_spike_quality'].fillna(1)
    )
    
    # Microstructure Gap Behavior
    data['daily_gap_size'] = data['open'] / data['close'].shift(1) - 1
    data['gap_closure_progress'] = (
        (data['close'] - np.minimum(data['open'], data['close'].shift(1))) /
        (np.maximum(data['open'], data['close'].shift(1)) - np.minimum(data['open'], data['close'].shift(1)))
    ).replace([np.inf, -np.inf], 0)
    
    # Price Rejection Patterns
    data['upper_shadow_rejection'] = (
        (data['high'] - np.maximum(data['open'], data['close'])) /
        (data['high'] - data['low']).replace(0, np.nan)
    )
    data['lower_shadow_support'] = (
        (np.minimum(data['open'], data['close']) - data['low']) /
        (data['high'] - data['low']).replace(0, np.nan)
    )
    
    # Opening Price Dynamics
    data['open_rejection_strength'] = (
        abs(data['close'] - data['open']) / 
        (data['high'] - data['low']).replace(0, np.nan)
    )
    data['overnight_gap_persistence'] = (
        np.sign(data['open'] - data['close'].shift(1)) * 
        np.sign(data['close'] - data['open'])
    )
    
    # Behavioral Range Constraints
    data['distance_from_20d_high'] = data['close'] / data['high'].rolling(window=20).max() - 1
    data['distance_from_20d_low'] = data['close'] / data['low'].rolling(window=20).min() - 1
    
    # Volatility Context
    data['atr_10d'] = (data['high'] - data['low']).rolling(window=10).mean()
    data['range_normalization'] = (data['high'] - data['low']) / data['atr_10d'].replace(0, np.nan)
    
    # Price Memory Effects
    data['close_level_attraction'] = (
        abs(data['close'] - data['close'].shift(1)) / 
        (data['high'] - data['low']).replace(0, np.nan)
    )
    data['open_close_consistency'] = (
        np.sign(data['close'] - data['open']) * 
        np.sign(data['close'].shift(1) - data['open'].shift(1))
    )
    
    # Divergence Synthesis
    # Trend-Volume Alignment
    data['trend_volume_alignment'] = (
        data['trend_persistence_score'].fillna(0) * 
        data['volume_confirmation_score'].fillna(0)
    )
    
    # Microstructure Adjustment
    data['microstructure_adjustment'] = (
        data['gap_closure_progress'].fillna(0.5) * 
        (1 - data['upper_shadow_rejection'].fillna(0))
    )
    
    # Range Constraints Application
    data['range_constraints'] = (
        (1 - abs(data['distance_from_20d_high'].fillna(0))) *
        data['daily_range_efficiency'].fillna(0)
    )
    
    # Volume Persistence Integration
    data['volume_persistence_integration'] = (
        data['volume_persistence'].fillna(0) * 
        np.sign(data['volume_trend'].fillna(1))
    )
    
    # Opening Dynamics Confirmation
    data['opening_dynamics'] = (
        data['open_rejection_strength'].fillna(0) * 
        data['overnight_gap_persistence'].fillna(0)
    )
    
    # Final Composite Factor
    data['composite_factor'] = (
        data['trend_volume_alignment'] *
        data['microstructure_adjustment'] *
        data['range_constraints'] *
        data['volume_persistence_integration'] *
        data['opening_dynamics']
    )
    
    return data['composite_factor']
