import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Timeframe Fractal Momentum Analysis
    # Short-Term Fractal Momentum (3-day)
    data['range_sum_3d'] = data['high'].rolling(window=3).apply(lambda x: np.sum(np.abs(x - data.loc[x.index, 'low'])), raw=False)
    data['close_path_3d'] = data['close'].diff().abs().rolling(window=2).sum()
    data['fractal_dim_3d'] = np.log(data['range_sum_3d']) / np.log(data['close_path_3d'].replace(0, np.nan))
    
    data['mom_2d'] = data['close'] / data['close'].shift(2) - 1
    data['mom_3d'] = data['close'] / data['close'].shift(3) - 1
    data['fractal_mom_short'] = data['fractal_dim_3d'] * (data['mom_2d'] - data['mom_3d'])
    
    # Medium-Term Fractal Momentum (8-day)
    data['range_sum_8d'] = data['high'].rolling(window=8).apply(lambda x: np.sum(np.abs(x - data.loc[x.index, 'low'])), raw=False)
    data['close_path_8d'] = data['close'].diff().abs().rolling(window=8).sum()
    data['fractal_dim_8d'] = np.log(data['range_sum_8d']) / np.log(data['close_path_8d'].replace(0, np.nan))
    
    data['mom_5d'] = data['close'] / data['close'].shift(5) - 1
    data['mom_8d'] = data['close'] / data['close'].shift(8) - 1
    data['fractal_mom_medium'] = data['fractal_dim_8d'] * (data['mom_5d'] - data['mom_8d'])
    
    # Fractal Momentum Acceleration
    data['fractal_mom_accel'] = data['fractal_mom_short'] - data['fractal_mom_medium']
    data['accel_magnitude'] = np.abs(data['fractal_mom_accel'])
    
    # Volume-Efficiency Breakout System
    # Efficiency Metrics
    data['basic_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['directional_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume Flow Dynamics
    data['volume_accel'] = (data['volume'] / data['volume'].shift(2)) - (data['volume'] / data['volume'].rolling(window=3).mean())
    data['volume_compression'] = data['volume'].rolling(window=5).std() / data['volume'].rolling(window=5).mean()
    data['volume_weighted_efficiency'] = data['basic_efficiency'] * data['volume_compression']
    
    # Breakout Detection
    data['upward_breakout'] = ((data['close'] > data['close'].shift(1)) & (data['high'] > data['high'].shift(1))).astype(float)
    data['downward_breakout'] = ((data['close'] < data['close'].shift(1)) & (data['low'] < data['low'].shift(1))).astype(float)
    data['breakout_strength'] = np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    
    data['volume_confirmed_breakout'] = data['breakout_strength'] * data['volume_accel']
    data['efficiency_confirmed_breakout'] = np.sign(data['directional_efficiency']) * data['volume_weighted_efficiency']
    
    # Range & Position Dynamics
    data['range_expansion'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    data['range_ratio'] = 1 - (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()) / (data['high'].rolling(window=10).max() - data['low'].rolling(window=10).min()).replace(0, np.nan)
    data['range_efficiency'] = data['basic_efficiency'] * data['range_ratio']
    
    data['position_strength'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Key Level Proximity
    data['dist_resistance'] = (data['high'].rolling(window=20).max() - data['close']) / data['close']
    data['dist_support'] = (data['close'] - data['low'].rolling(window=20).min()) / data['close']
    data['key_level_multiplier'] = np.minimum(data['dist_support'], data['dist_resistance'])
    
    # Contextual Multipliers
    data['range_efficiency_multiplier'] = data['range_efficiency']
    data['position_strength_multiplier'] = data['position_strength']
    
    # Synthesize Adaptive Alpha Factor
    # Core Fractal Momentum Signal
    core_signal = data['fractal_mom_accel'] * data['accel_magnitude']
    
    # Volume-Efficiency Breakout Integration
    volume_breakout_component = core_signal * data['volume_confirmed_breakout'] * data['efficiency_confirmed_breakout']
    
    # Range & Position Context Application
    final_factor = volume_breakout_component * data['range_efficiency_multiplier'] * data['position_strength_multiplier'] * data['key_level_multiplier']
    
    return final_factor
