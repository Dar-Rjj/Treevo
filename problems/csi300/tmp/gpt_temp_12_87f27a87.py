import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Asymmetric Microstructure Pressure Divergence factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Initialize pressure components
    data['up_pressure'] = 0.0
    data['down_pressure'] = 0.0
    
    # Calculate basic price movements
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['is_up'] = (data['price_change'] > 0).astype(int)
    data['is_down'] = (data['price_change'] < 0).astype(int)
    
    # Calculate consecutive up/down sequences
    data['consecutive_up'] = 0
    data['consecutive_down'] = 0
    data['up_volume_accum'] = 0.0
    data['down_volume_accum'] = 0.0
    
    # Calculate consecutive sequences and volume accumulation
    for i in range(1, len(data)):
        if data['is_up'].iloc[i] == 1:
            data.loc[data.index[i], 'consecutive_up'] = data['consecutive_up'].iloc[i-1] + 1
            data.loc[data.index[i], 'consecutive_down'] = 0
            data.loc[data.index[i], 'up_volume_accum'] = data['up_volume_accum'].iloc[i-1] + data['volume'].iloc[i]
            data.loc[data.index[i], 'down_volume_accum'] = 0
        elif data['is_down'].iloc[i] == 1:
            data.loc[data.index[i], 'consecutive_down'] = data['consecutive_down'].iloc[i-1] + 1
            data.loc[data.index[i], 'consecutive_up'] = 0
            data.loc[data.index[i], 'down_volume_accum'] = data['down_volume_accum'].iloc[i-1] + data['volume'].iloc[i]
            data.loc[data.index[i], 'up_volume_accum'] = 0
        else:
            data.loc[data.index[i], 'consecutive_up'] = 0
            data.loc[data.index[i], 'consecutive_down'] = 0
            data.loc[data.index[i], 'up_volume_accum'] = 0
            data.loc[data.index[i], 'down_volume_accum'] = 0
    
    # Calculate volatility ratios
    data['up_vol_ratio'] = (data['high'] - data['open']) / (data['open'] - data['low'] + 1e-8)
    data['down_vol_ratio'] = (data['open'] - data['low']) / (data['high'] - data['open'] + 1e-8)
    
    # Calculate move efficiency
    data['up_move_efficiency'] = np.where(
        data['price_change'] > 0,
        data['volume'] / (data['price_change'] + 1e-8),
        0
    )
    data['down_move_efficiency'] = np.where(
        data['price_change'] < 0,
        data['volume'] / (-data['price_change'] + 1e-8),
        0
    )
    
    # Calculate Upward Pressure Intensity
    up_momentum = data['consecutive_up'] * data['up_volume_accum']
    up_efficiency = data['up_move_efficiency'] * data['up_vol_ratio']
    data['up_pressure'] = up_momentum * up_efficiency
    
    # Calculate Downward Pressure Intensity  
    down_momentum = data['consecutive_down'] * data['down_volume_accum']
    down_efficiency = data['down_move_efficiency'] * data['down_vol_ratio']
    data['down_pressure'] = down_momentum * down_efficiency
    
    # Calculate Pressure Acceleration
    data['up_pressure_accel'] = (data['up_pressure'] - data['up_pressure'].shift(1)) - \
                               (data['up_pressure'].shift(1) - data['up_pressure'].shift(2))
    data['down_pressure_accel'] = (data['down_pressure'] - data['down_pressure'].shift(1)) - \
                                 (data['down_pressure'].shift(1) - data['down_pressure'].shift(2))
    
    # Calculate efficiency changes
    data['up_eff_change'] = data['up_move_efficiency'] - data['up_move_efficiency'].shift(1)
    data['down_eff_change'] = data['down_move_efficiency'] - data['down_move_efficiency'].shift(1)
    
    # Pressure Direction Divergence
    pos_up_neg_down = (data['up_pressure_accel'] > 0) & (data['down_pressure_accel'] < 0)
    neg_up_pos_down = (data['up_pressure_accel'] < 0) & (data['down_pressure_accel'] > 0)
    direction_divergence = pos_up_neg_down.astype(int) - neg_up_pos_down.astype(int)
    
    # Efficiency-Volatility Divergence
    high_eff_low_vol = (data['up_move_efficiency'] > data['up_move_efficiency'].rolling(10).mean()) & \
                      (data['up_vol_ratio'] < data['up_vol_ratio'].rolling(10).mean())
    low_eff_high_vol = (data['up_move_efficiency'] < data['up_move_efficiency'].rolling(10).mean()) & \
                      (data['up_vol_ratio'] > data['up_vol_ratio'].rolling(10).mean())
    efficiency_divergence = high_eff_low_vol.astype(int) - low_eff_high_vol.astype(int)
    
    # Multi-Timeframe Confirmation
    # Short-term vs medium-term acceleration
    up_accel_3d = data['up_pressure_accel'].rolling(3).mean()
    up_accel_10d = data['up_pressure_accel'].rolling(10).mean()
    down_accel_3d = data['down_pressure_accel'].rolling(3).mean()
    down_accel_10d = data['down_pressure_accel'].rolling(10).mean()
    
    timeframe_alignment = ((up_accel_3d * up_accel_10d > 0) & (down_accel_3d * down_accel_10d > 0)).astype(int)
    
    # Divergence Persistence
    sustained_asymmetry = (data['up_pressure'].rolling(5).mean() > data['down_pressure'].rolling(5).mean()).astype(int) - \
                         (data['up_pressure'].rolling(5).mean() < data['down_pressure'].rolling(5).mean()).astype(int)
    
    # Combine all components into final factor
    factor = (direction_divergence * 0.4 + 
              efficiency_divergence * 0.3 + 
              timeframe_alignment * 0.2 + 
              sustained_asymmetry * 0.1)
    
    # Normalize the factor
    factor = (factor - factor.rolling(20).mean()) / (factor.rolling(20).std() + 1e-8)
    
    return factor
