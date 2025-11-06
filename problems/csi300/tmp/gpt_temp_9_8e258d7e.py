import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Asymmetric Price-Volume Relationships
    # Upside-Downside Volume Ratio
    up_days = data['close'] > data['close'].shift(1)
    down_days = data['close'] < data['close'].shift(1)
    
    # Rolling upside and downside volume
    up_volume_rolling = data['volume'].rolling(window=5).apply(
        lambda x: np.nanmean([x.iloc[i] for i in range(len(x)) if i > 0 and x.index[i-1] in data.index and data.loc[x.index[i-1], 'close'] < data.loc[x.index[i], 'close']] or [np.nan]), 
        raw=False
    )
    down_volume_rolling = data['volume'].rolling(window=5).apply(
        lambda x: np.nanmean([x.iloc[i] for i in range(len(x)) if i > 0 and x.index[i-1] in data.index and data.loc[x.index[i-1], 'close'] > data.loc[x.index[i], 'close']] or [np.nan]), 
        raw=False
    )
    
    upside_downside_ratio = up_volume_rolling / down_volume_rolling
    
    # Price Gap Volume Absorption
    price_range_volume_t = (data['high'] - data['low']) / data['volume']
    price_range_volume_t1 = (data['high'].shift(1) - data['low'].shift(1)) / data['volume'].shift(1)
    price_gap_volume_absorption = price_range_volume_t - price_range_volume_t1
    
    # Extreme Move Validation
    price_move = (data['close'] - data['open']) / data['open']
    volume_change_sign = np.sign(data['volume'] - data['volume'].shift(1))
    extreme_move_validation = price_move * volume_change_sign
    
    # Dynamic Support-Resistance Framework
    # Recent Price Extremes
    dynamic_resistance = data['high'].rolling(window=5).max()
    dynamic_support = data['low'].rolling(window=5).min()
    
    # Price Position Relative to Extremes
    resistance_proximity = (data['high'] - dynamic_resistance) / (data['high'] - data['low'])
    support_proximity = (dynamic_support - data['low']) / (data['high'] - data['low'])
    
    # Volume Confirmation at Extremes
    high_volume_at_resistance = data['volume'] * (data['close'] > 0.8 * dynamic_resistance)
    high_volume_at_support = data['volume'] * (data['close'] < 1.2 * dynamic_support)
    
    # Multi-Timeframe Momentum Divergence
    # Short vs Medium Momentum
    short_term_momentum = (data['close'] / data['close'].shift(1) - 1)
    medium_term_momentum = (data['close'] / data['close'].shift(3) - 1)
    momentum_divergence = short_term_momentum - medium_term_momentum
    
    # Volume-Weighted Acceleration
    volume_weighted_return = (data['close'] - data['close'].shift(1)) * data['volume']
    volume_weighted_acceleration = volume_weighted_return - volume_weighted_return.shift(1)
    
    # Adaptive Composite Scoring
    # Core Asymmetry Score
    core_asymmetry_score = upside_downside_ratio * price_gap_volume_absorption
    
    # Extremes Pressure Score
    extremes_pressure_score = (resistance_proximity - support_proximity) * extreme_move_validation
    
    # Dynamic Momentum Score with volume multipliers
    volume_ratio = data['volume'] / data['volume'].shift(1)
    high_volume_multiplier = np.where(volume_ratio > 1.2, 1.5, 1.0)
    low_volume_multiplier = np.where(volume_ratio < 0.8, 0.7, 1.0)
    volume_multiplier = high_volume_multiplier * low_volume_multiplier
    
    dynamic_momentum_score = momentum_divergence * volume_weighted_acceleration * volume_multiplier
    
    # Final composite factor
    composite_factor = (
        core_asymmetry_score.fillna(0) + 
        extremes_pressure_score.fillna(0) + 
        dynamic_momentum_score.fillna(0)
    )
    
    return pd.Series(composite_factor, index=data.index, name='factor')
