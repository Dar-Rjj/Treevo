import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Volatility Structure
    # Ultra-Short Volatility
    data['high_roll_2'] = data['high'].rolling(window=3, min_periods=1).max()
    data['low_roll_2'] = data['low'].rolling(window=3, min_periods=1).min()
    data['ultra_short_vol'] = (data['high'] - data['low']) / (data['high_roll_2'] - data['low_roll_2'] + 1e-6)
    
    # Short-Term Volatility
    data['high_roll_5'] = data['high'].rolling(window=5, min_periods=1).max()
    data['low_roll_5'] = data['low'].rolling(window=5, min_periods=1).min()
    data['high_roll_10'] = data['high'].rolling(window=10, min_periods=1).max()
    data['low_roll_10'] = data['low'].rolling(window=10, min_periods=1).min()
    data['short_term_vol'] = (data['high_roll_5'] - data['low_roll_5']) / (data['high_roll_10'] - data['low_roll_10'] + 1e-6)
    
    # Medium-Term Volatility
    data['high_roll_20'] = data['high'].rolling(window=20, min_periods=1).max()
    data['low_roll_20'] = data['low'].rolling(window=20, min_periods=1).min()
    data['medium_term_vol'] = (data['high_roll_10'] - data['low_roll_10']) / (data['high_roll_20'] - data['low_roll_20'] + 1e-6)
    
    # Asymmetric Price Behavior Components
    # Calculate up-move and down-move components for rolling window
    up_move = []
    down_move = []
    
    for i in range(len(data)):
        if i < 3:
            up_move.append(0)
            down_move.append(0)
            continue
            
        up_sum = 0
        down_sum = 0
        for j in range(i-3, i+1):
            high_low_range = data['high'].iloc[j] - data['low'].iloc[j]
            if high_low_range > 0:
                up_move_component = (data['high'].iloc[j] - max(data['open'].iloc[j], data['close'].iloc[j])) / high_low_range
                down_move_component = (min(data['open'].iloc[j], data['close'].iloc[j]) - data['low'].iloc[j]) / high_low_range
                up_sum += up_move_component
                down_sum += down_move_component
        
        up_move.append(up_sum)
        down_move.append(down_sum)
    
    data['up_move_concentration'] = up_move
    data['down_move_concentration'] = down_move
    data['asymmetry_ratio'] = (data['up_move_concentration'] - data['down_move_concentration']) / (data['up_move_concentration'] + data['down_move_concentration'] + 1e-6)
    
    # Volume-Price Interaction Dynamics
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_concentration'] = data['volume'] / (data['volume_ma_5'] + 1e-6)
    
    data['price_impact_efficiency'] = abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-6)
    
    # Volume-Price Asymmetry
    volume_up = []
    volume_down = []
    total_volume = []
    
    for i in range(len(data)):
        if i < 2:
            volume_up.append(0)
            volume_down.append(0)
            total_volume.append(0)
            continue
            
        up_sum = 0
        down_sum = 0
        total_sum = 0
        for j in range(i-2, i+1):
            if data['close'].iloc[j] > data['open'].iloc[j]:
                up_sum += data['volume'].iloc[j]
            elif data['close'].iloc[j] < data['open'].iloc[j]:
                down_sum += data['volume'].iloc[j]
            total_sum += data['volume'].iloc[j]
        
        volume_up.append(up_sum)
        volume_down.append(down_sum)
        total_volume.append(total_sum)
    
    data['volume_up_sum'] = volume_up
    data['volume_down_sum'] = volume_down
    data['volume_total_sum'] = total_volume
    data['volume_price_asymmetry'] = (data['volume_up_sum'] - data['volume_down_sum']) / (data['volume_total_sum'] + 1e-6)
    
    # Multi-Scale Momentum Components
    data['ultra_short_momentum'] = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-6)
    
    # Short-Term Momentum
    data['high_roll_3'] = data['high'].rolling(window=3, min_periods=1).max()
    data['low_roll_3'] = data['low'].rolling(window=3, min_periods=1).min()
    data['short_term_momentum'] = (data['close'] - data['close'].shift(3)) / (data['high_roll_3'] - data['low_roll_3'] + 1e-6)
    
    # Medium-Term Momentum
    data['high_roll_6'] = data['high'].rolling(window=6, min_periods=1).max()
    data['low_roll_6'] = data['low'].rolling(window=6, min_periods=1).min()
    data['medium_term_momentum'] = (data['close'] - data['close'].shift(6)) / (data['high_roll_6'] - data['low_roll_6'] + 1e-6)
    
    # Core Asymmetry Signals
    data['volatility_asymmetry'] = data['ultra_short_vol'] * data['asymmetry_ratio'] * np.sign(data['close'] - data['open'])
    
    data['volume_price_asymmetry_signal'] = data['volume_price_asymmetry'] * data['price_impact_efficiency'] * data['volume_concentration']
    
    data['multi_scale_momentum_asymmetry'] = data['ultra_short_momentum'] * data['short_term_momentum'] * data['medium_term_momentum']
    
    # Hierarchical Factor Composition
    data['base_asymmetry_factor'] = data['volatility_asymmetry'] * data['volume_price_asymmetry_signal']
    
    # Momentum-Enhanced Factor
    momentum_condition_1 = np.where((data['multi_scale_momentum_asymmetry'] > 0) & (data['asymmetry_ratio'] > 0.1), 1.3, 1.0)
    momentum_condition_2 = np.where((data['short_term_vol'] > 0.8) & (data['volume_concentration'] > 1.1), 1.2, 1.0)
    momentum_condition_3 = np.where((data['medium_term_vol'] < 0.5) & (data['volume_price_asymmetry'] < -0.1), 0.8, 1.0)
    
    data['momentum_enhanced_factor'] = data['base_asymmetry_factor'] * momentum_condition_1 * momentum_condition_2 * momentum_condition_3
    
    # Final Hierarchical Factor
    final_condition_1 = np.where((data['up_move_concentration'] > 0.6) & (data['ultra_short_momentum'] > 0), 1.1, 1.0)
    final_condition_2 = np.where((data['down_move_concentration'] > 0.6) & (data['price_impact_efficiency'] < 0.3), 0.9, 1.0)
    
    data['final_hierarchical_factor'] = data['momentum_enhanced_factor'] * final_condition_1 * final_condition_2
    
    return data['final_hierarchical_factor']
