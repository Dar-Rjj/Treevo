import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Momentum Acceleration components
    data['price_momentum'] = data['close'] - data['close'].shift(5)
    
    # Fractal Acceleration
    close_diff_1 = data['close'] - data['close'].shift(1)
    close_diff_2 = data['close'].shift(1) - data['close'].shift(2)
    denominator = np.abs(data['close'].shift(1) - data['close'].shift(2))
    denominator = np.where(denominator == 0, 1e-6, denominator)  # Avoid division by zero
    data['fractal_acceleration'] = (close_diff_1 - close_diff_2) / denominator
    
    # Range Adjustment
    daily_range = data['high'] - data['low']
    daily_range = np.where(daily_range == 0, 1e-6, daily_range)  # Avoid division by zero
    data['range_adjustment'] = data['fractal_acceleration'] / daily_range
    
    # Volume Synchronization components
    # Volume Persistence - count of consecutive days with increasing volume
    volume_increase = (data['volume'] > data['volume'].shift(1)).astype(int)
    data['volume_persistence'] = volume_increase.groupby(volume_increase.index).expanding().apply(
        lambda x: (x == 1).cumsum().iloc[-1] if (x == 1).any() else 0
    ).reset_index(level=0, drop=True)
    
    # Volume Efficiency
    range_1 = data['high'] - data['low']
    range_2 = np.abs(data['high'] - data['close'].shift(1))
    range_3 = np.abs(data['low'] - data['close'].shift(1))
    effective_range = np.maximum(range_1, np.maximum(range_2, range_3))
    effective_range = np.where(effective_range == 0, 1e-6, effective_range)  # Avoid division by zero
    data['volume_efficiency'] = data['volume'] / effective_range
    
    # Fractal Range Movement components
    # Price Efficiency
    data['price_efficiency'] = (data['close'] - data['open']) / daily_range
    
    # Range Direction
    data['price_direction'] = np.sign(data['close'] - data['close'].shift(1))
    
    # Calculate rolling max high and min low over 3-day window
    rolling_max_high = data['high'].rolling(window=3, min_periods=1).max()
    rolling_min_low = data['low'].rolling(window=3, min_periods=1).min()
    data['range_direction'] = data['price_direction'] * (rolling_max_high - rolling_min_low)
    
    # Fractal Pressure components
    # Morning Pressure
    data['morning_pressure'] = (data['high'] - data['open']) * (data['close'] - data['low'])
    
    # Afternoon Pressure
    data['afternoon_pressure'] = (data['open'] - data['low']) * (data['high'] - data['close'])
    
    # Pressure Differential
    data['pressure_differential'] = data['morning_pressure'] - data['afternoon_pressure']
    
    # Alpha Generation
    # Short-term Core
    data['short_term_core'] = (data['range_adjustment'] * data['price_efficiency']) * data['volume_persistence']
    
    # Medium-term Core
    data['medium_term_core'] = (data['fractal_acceleration'] * data['range_direction']) * data['volume_efficiency']
    
    # Final Alpha
    data['alpha'] = data['short_term_core'] * data['medium_term_core'] * data['pressure_differential']
    
    return data['alpha']
