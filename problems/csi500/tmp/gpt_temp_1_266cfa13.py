import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum-Volume Regime Factor
    Combines momentum across intraday, short-term, and medium-term timeframes
    with volume persistence analysis and volatility regime assessment
    """
    data = df.copy()
    
    # Multi-Timeframe Momentum Extraction
    # Intraday Momentum
    data['intraday_return'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['intraday_direction'] = np.sign(data['close'] - data['open'])
    data['intraday_strength'] = np.abs(data['intraday_return'])
    
    # Short-term Momentum (3-day)
    data['close_3d_ago'] = data['close'].shift(3)
    data['high_3d_max'] = data['high'].rolling(window=4, min_periods=4).max()
    data['low_3d_min'] = data['low'].rolling(window=4, min_periods=4).min()
    data['short_momentum'] = (data['close'] - data['close_3d_ago']) / (data['high_3d_max'] - data['low_3d_min'] + 1e-8)
    data['short_direction'] = np.sign(data['close'] - data['close_3d_ago'])
    
    # Calculate momentum persistence for short-term
    data['short_dir_change'] = data['short_direction'] != data['short_direction'].shift(1)
    data['short_persistence'] = data.groupby(data['short_dir_change'].cumsum())['short_direction'].cumcount() + 1
    
    # Medium-term Momentum (10-day)
    data['close_10d_ago'] = data['close'].shift(10)
    data['high_10d_max'] = data['high'].rolling(window=11, min_periods=11).max()
    data['low_10d_min'] = data['low'].rolling(window=11, min_periods=11).min()
    data['medium_momentum'] = (data['close'] - data['close_10d_ago']) / (data['high_10d_max'] - data['low_10d_min'] + 1e-8)
    data['medium_direction'] = np.sign(data['close'] - data['close_10d_ago'])
    data['medium_strength'] = np.abs(data['medium_momentum'])
    
    # Volume Persistence Analysis
    data['volume_ratio'] = data['volume'] / (data['volume'].shift(1) + 1e-8)
    data['volume_direction'] = np.sign(data['volume'] - data['volume'].shift(1))
    data['volume_strength'] = np.abs(data['volume_ratio'] - 1)
    
    # Volume persistence tracking
    data['volume_increase'] = data['volume'] > data['volume'].shift(1)
    data['volume_decrease'] = data['volume'] < data['volume'].shift(1)
    
    # Consecutive volume increases
    data['vol_inc_change'] = data['volume_increase'] != data['volume_increase'].shift(1)
    data['consecutive_vol_inc'] = data.groupby(data['vol_inc_change'].cumsum())['volume_increase'].cumcount() + 1
    data['consecutive_vol_inc'] = data['consecutive_vol_inc'].where(data['volume_increase'], 0)
    
    # Consecutive volume decreases
    data['vol_dec_change'] = data['volume_decrease'] != data['volume_decrease'].shift(1)
    data['consecutive_vol_dec'] = data.groupby(data['vol_dec_change'].cumsum())['volume_decrease'].cumcount() + 1
    data['consecutive_vol_dec'] = data['consecutive_vol_dec'].where(data['volume_decrease'], 0)
    
    # Volume-Price Alignment
    data['direction_alignment'] = data['volume_direction'] * data['intraday_direction']
    data['strength_alignment'] = data['volume_strength'] * data['intraday_strength']
    
    # Multi-timeframe alignment check
    data['multi_alignment'] = (
        (data['volume_direction'] * data['intraday_direction'] > 0).astype(int) +
        (data['volume_direction'] * data['short_direction'] > 0).astype(int) +
        (data['volume_direction'] * data['medium_direction'] > 0).astype(int)
    )
    
    # Volatility Regime Assessment
    data['daily_range'] = data['high'] - data['low']
    data['range_ratio'] = data['daily_range'] / (data['daily_range'].shift(1) + 1e-8)
    data['range_direction'] = np.sign(data['daily_range'] - data['daily_range'].shift(1))
    
    # Volatility persistence
    data['range_3d_avg'] = data['daily_range'].rolling(window=3, min_periods=3).mean()
    data['range_5d_avg'] = data['daily_range'].rolling(window=5, min_periods=5).mean()
    data['range_trend'] = np.sign(data['range_3d_avg'] - data['range_5d_avg'])
    
    # Volatility regime classification
    data['volatility_regime'] = np.where(
        data['daily_range'] > data['daily_range'].rolling(window=20, min_periods=20).quantile(0.7),
        2,  # High volatility
        np.where(data['daily_range'] < data['daily_range'].rolling(window=20, min_periods=20).quantile(0.3), 
                0,  # Low volatility
                1)  # Normal volatility
    )
    
    # Volatility-Volume Relationship
    data['volume_per_volatility'] = data['volume'] / (data['daily_range'] + 1e-8)
    data['volume_volatility_alignment'] = data['volume_direction'] * data['range_direction']
    data['efficiency_measure'] = np.abs(data['close'] - data['open']) / (data['daily_range'] + 1e-8)
    
    # Adaptive Signal Blending
    # Volume persistence bonus
    data['volume_persistence_bonus'] = np.maximum(
        data['consecutive_vol_inc'], data['consecutive_vol_dec']
    ) * 0.05
    
    # Momentum persistence bonus
    data['momentum_persistence_bonus'] = data['short_persistence'] * 0.03
    
    # Volatility regime adjustment
    data['volatility_regime_adjustment'] = np.where(
        data['volatility_regime'] == 2, 0.8,  # High volatility: reduce weight
        np.where(data['volatility_regime'] == 0, 1.2, 1.0)  # Low volatility: increase weight
    )
    
    # Timeframe weights
    data['intraday_weight'] = 0.4 * (1 + data['volume_persistence_bonus'])
    data['short_term_weight'] = 0.35 * (1 + data['momentum_persistence_bonus'])
    data['medium_term_weight'] = 0.25 * data['volatility_regime_adjustment']
    
    # Volume confirmation scaling
    data['alignment_score'] = data['multi_alignment'] / 3.0
    data['positive_alignment_multiplier'] = 1 + (data['volume_strength'] * data['alignment_score'])
    data['negative_alignment_penalty'] = 1 - (data['volume_strength'] * (1 - data['alignment_score']))
    
    # Consecutive aligned days
    data['aligned_days_change'] = (data['direction_alignment'] > 0) != (data['direction_alignment'].shift(1) > 0)
    data['consecutive_aligned_days'] = data.groupby(data['aligned_days_change'].cumsum())['direction_alignment'].transform(
        lambda x: (x > 0).cumsum()
    )
    data['persistence_bonus'] = 1 + (data['consecutive_aligned_days'] * 0.1)
    
    # Volatility context adjustment
    data['excess_volatility'] = np.maximum(0, data['daily_range'] / data['daily_range'].rolling(window=20, min_periods=20).mean() - 1)
    data['high_vol_dampening'] = 1 / (1 + data['excess_volatility'])
    data['low_vol_amplification'] = 1 + data['efficiency_measure']
    
    # Final Factor Construction
    # Weighted momentum score
    data['weighted_momentum'] = (
        data['intraday_return'] * data['intraday_weight'] +
        data['short_momentum'] * data['short_term_weight'] +
        data['medium_momentum'] * data['medium_term_weight']
    )
    
    # Apply volume scaling
    data['volume_scaled_factor'] = data['weighted_momentum'] * data['positive_alignment_multiplier']
    data['volume_scaled_factor'] = np.where(
        data['alignment_score'] < 0.5,
        data['volume_scaled_factor'] * data['negative_alignment_penalty'],
        data['volume_scaled_factor']
    )
    
    # Apply persistence bonus
    data['persistence_enhanced'] = data['volume_scaled_factor'] * data['persistence_bonus']
    
    # Apply volatility adjustment
    data['volatility_adjusted'] = np.where(
        data['volatility_regime'] == 2,
        data['persistence_enhanced'] * data['high_vol_dampening'],
        np.where(data['volatility_regime'] == 0,
                data['persistence_enhanced'] * data['low_vol_amplification'],
                data['persistence_enhanced'])
    )
    
    # Multi-timeframe direction consistency bonus
    direction_consistency = (
        (data['intraday_direction'] == data['short_direction']).astype(int) +
        (data['intraday_direction'] == data['medium_direction']).astype(int) +
        (data['short_direction'] == data['medium_direction']).astype(int)
    ) / 3.0
    
    data['final_factor'] = data['volatility_adjusted'] * (1 + direction_consistency * 0.2)
    
    return data['final_factor']
