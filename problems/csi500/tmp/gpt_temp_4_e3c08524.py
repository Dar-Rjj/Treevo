import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum-Volume Convergence factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Core Momentum Components
    # Multi-Timeframe Price Momentum
    data['ultra_short_return'] = data['close'] - data['close'].shift(1)
    data['intraday_strength'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Short-term (3-day) momentum
    data['short_return'] = data['close'] - data['close'].shift(2)
    data['short_avg_range'] = (
        (data['high'] - data['low']) + 
        (data['high'].shift(1) - data['low'].shift(1)) + 
        (data['high'].shift(2) - data['low'].shift(2))
    ) / 3
    
    # Medium-term (5-day) momentum
    data['medium_return'] = data['close'] - data['close'].shift(4)
    data['medium_avg_range'] = (
        (data['high'] - data['low']) + 
        (data['high'].shift(1) - data['low'].shift(1)) + 
        (data['high'].shift(2) - data['low'].shift(2)) + 
        (data['high'].shift(3) - data['low'].shift(3)) + 
        (data['high'].shift(4) - data['low'].shift(4))
    ) / 5
    
    # Volatility-Scaled Momentum
    data['short_vsm'] = data['short_return'] / (
        (data['high'] - data['low']) + 
        (data['high'].shift(1) - data['low'].shift(1)) + 
        (data['high'].shift(2) - data['low'].shift(2))
    ).replace(0, np.nan)
    
    data['medium_vsm'] = data['medium_return'] / (
        (data['high'] - data['low']) + 
        (data['high'].shift(1) - data['low'].shift(1)) + 
        (data['high'].shift(2) - data['low'].shift(2)) + 
        (data['high'].shift(3) - data['low'].shift(3)) + 
        (data['high'].shift(4) - data['low'].shift(4))
    ).replace(0, np.nan)
    
    data['volatility_regime_indicator'] = data['short_vsm'] / data['medium_vsm'].replace(0, np.nan)
    
    # Volume Dynamics
    # Volume Trend Components
    data['volume_direction'] = np.sign(data['volume'] - data['volume'].shift(1))
    
    # Calculate volume persistence (consecutive days with same direction)
    volume_persistence = []
    current_streak = 0
    current_direction = 0
    
    for i in range(len(data)):
        if i == 0 or pd.isna(data['volume_direction'].iloc[i]):
            volume_persistence.append(0)
            current_streak = 0
            current_direction = 0
        elif data['volume_direction'].iloc[i] == current_direction:
            current_streak += 1
            volume_persistence.append(current_streak)
        else:
            current_streak = 1
            current_direction = data['volume_direction'].iloc[i]
            volume_persistence.append(current_streak)
    
    data['volume_persistence'] = volume_persistence
    data['volume_momentum'] = data['volume'] / data['volume'].shift(1).replace(0, np.nan) - 1
    
    # Volume-Price Alignment
    data['direction_alignment'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Calculate alignment persistence
    alignment_persistence = []
    current_alignment_streak = 0
    
    for i in range(len(data)):
        if i == 0 or pd.isna(data['direction_alignment'].iloc[i]):
            alignment_persistence.append(0)
            current_alignment_streak = 0
        elif data['direction_alignment'].iloc[i] > 0:
            current_alignment_streak += 1
            alignment_persistence.append(current_alignment_streak)
        else:
            current_alignment_streak = 0
            alignment_persistence.append(0)
    
    data['alignment_persistence'] = alignment_persistence
    data['alignment_strength'] = data['alignment_persistence'] * abs(data['close'] - data['close'].shift(1))
    
    # Volume Regime Detection
    data['volume_ratio'] = (
        (data['volume'] + data['volume'].shift(1) + data['volume'].shift(2)) / 
        (data['volume'].shift(3) + data['volume'].shift(4) + data['volume'].shift(5))
    ).replace(0, np.nan)
    
    # Regime-Adaptive Blending
    factor_values = []
    
    for i in range(len(data)):
        if i < 5 or any(pd.isna(data.iloc[i][['short_vsm', 'medium_vsm', 'volume_persistence', 'alignment_strength', 'volume_ratio']])):
            factor_values.append(np.nan)
            continue
            
        # Volatility Regime Weights
        vol_regime = data['volatility_regime_indicator'].iloc[i]
        
        if vol_regime > 1.1:  # High Volatility
            momentum_weight = 0.4 * data['short_vsm'].iloc[i]
            volume_weight = 0.6 * data['volume_persistence'].iloc[i]
            alignment_weight = 0.8 * data['alignment_strength'].iloc[i]
        elif vol_regime >= 0.9:  # Normal Volatility
            momentum_weight = 0.6 * data['medium_vsm'].iloc[i]
            volume_weight = 0.4 * data['volume_persistence'].iloc[i]
            alignment_weight = 1.0 * data['alignment_strength'].iloc[i]
        else:  # Low Volatility
            momentum_weight = 0.8 * data['medium_vsm'].iloc[i]
            volume_weight = 0.2 * data['volume_persistence'].iloc[i]
            alignment_weight = 1.2 * data['alignment_strength'].iloc[i]
        
        # Volume Regime Multipliers
        vol_ratio = data['volume_ratio'].iloc[i]
        
        if vol_ratio > 1.1:  # High Volume
            volume_multiplier = 1.3
        elif vol_ratio >= 0.9:  # Normal Volume
            volume_multiplier = 1.0
        else:  # Low Volume
            volume_multiplier = 0.7
        
        # Convergence Signal
        base_signal = momentum_weight + volume_weight + alignment_weight
        regime_adjusted = base_signal * volume_multiplier
        
        # Persistence Boost
        persistence_boost = 1 + min(data['volume_persistence'].iloc[i], 5) / 10
        final_factor = regime_adjusted * persistence_boost
        
        factor_values.append(final_factor)
    
    # Create output series
    factor_series = pd.Series(factor_values, index=data.index, name='regime_adaptive_momentum_volume')
    
    return factor_series
