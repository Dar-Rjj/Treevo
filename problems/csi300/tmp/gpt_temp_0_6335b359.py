import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Volatility Asymmetry Ratio
    high_low_range = data['high'] - data['low']
    close_open_range = data['close'] - data['open']
    
    # Avoid division by zero
    close_open_range_adj = np.where(np.abs(close_open_range) < 1e-8, np.sign(close_open_range) * 1e-8, close_open_range)
    high_low_range_adj = np.where(high_low_range < 1e-8, 1e-8, high_low_range)
    
    close_low_dist = data['close'] - data['low']
    high_close_dist = data['high'] - data['close']
    
    volatility_asymmetry_ratio = ((high_low_range / close_open_range_adj) * 
                                 ((close_low_dist - high_close_dist) / high_low_range_adj)) / \
                                ((high_low_range / close_open_range_adj) * 
                                 ((high_close_dist - close_low_dist) / high_low_range_adj) + 1e-8)
    
    # Volatility Asymmetry Persistence
    var_ratio = volatility_asymmetry_ratio
    var_sign_changes = pd.Series(np.zeros(len(data)), index=data.index)
    
    for i in range(2, len(data)):
        if i >= 3:
            current_sign = np.sign(var_ratio.iloc[i] / var_ratio.iloc[i-1] - 1)
            prev_sign = np.sign(var_ratio.iloc[i-1] / var_ratio.iloc[i-2] - 1)
            prev_prev_sign = np.sign(var_ratio.iloc[i-2] / var_ratio.iloc[i-3] - 1)
            
            persistence_count = 0
            if current_sign == prev_sign:
                persistence_count += 1
            if prev_sign == prev_prev_sign:
                persistence_count += 1
            if i >= 4 and prev_prev_sign == np.sign(var_ratio.iloc[i-3] / var_ratio.iloc[i-4] - 1):
                persistence_count += 1
            
            var_sign_changes.iloc[i] = persistence_count / 3
    
    volatility_asymmetry_persistence = var_sign_changes
    
    # Pressure Asymmetry
    pressure_asymmetry = ((data['close'] - data['low']) - (data['high'] - data['close'])) / high_low_range_adj
    
    # Pressure Asymmetry Persistence
    pa_sign_changes = pd.Series(np.zeros(len(data)), index=data.index)
    
    for i in range(2, len(data)):
        if i >= 3:
            current_sign = np.sign(pressure_asymmetry.iloc[i])
            prev_sign = np.sign(pressure_asymmetry.iloc[i-1])
            prev_prev_sign = np.sign(pressure_asymmetry.iloc[i-2])
            
            persistence_count = 0
            if current_sign == prev_sign:
                persistence_count += 1
            if prev_sign == prev_prev_sign:
                persistence_count += 1
            if i >= 4 and prev_prev_sign == np.sign(pressure_asymmetry.iloc[i-3]):
                persistence_count += 1
            
            pa_sign_changes.iloc[i] = persistence_count / 3
    
    pressure_asymmetry_persistence = pa_sign_changes
    
    # Volume Asymmetry Divergence
    volume_asymmetry_divergence = (data['volume'] * high_low_range * 
                                  ((data['close'] - data['low']) - (data['high'] - data['close'])) / high_low_range_adj -
                                  data['volume'] * close_open_range * 
                                  ((data['high'] - data['close']) - (data['close'] - data['low'])) / high_low_range_adj)
    
    # Volume Asymmetry Timing
    vad_sign_changes = pd.Series(np.zeros(len(data)), index=data.index)
    
    for i in range(2, len(data)):
        if i >= 3:
            current_sign = np.sign(volume_asymmetry_divergence.iloc[i] / volume_asymmetry_divergence.iloc[i-1] - 1)
            prev_sign = np.sign(volume_asymmetry_divergence.iloc[i-1] / volume_asymmetry_divergence.iloc[i-2] - 1)
            prev_prev_sign = np.sign(volume_asymmetry_divergence.iloc[i-2] / volume_asymmetry_divergence.iloc[i-3] - 1)
            
            persistence_count = 0
            if current_sign == prev_sign:
                persistence_count += 1
            if prev_sign == prev_prev_sign:
                persistence_count += 1
            if i >= 4 and prev_prev_sign == np.sign(volume_asymmetry_divergence.iloc[i-3] / volume_asymmetry_divergence.iloc[i-4] - 1):
                persistence_count += 1
            
            vad_sign_changes.iloc[i] = persistence_count / 3
    
    volume_asymmetry_timing = vad_sign_changes
    
    # Gap Asymmetry (computed but not used in final composite)
    gap_asymmetry = ((data['open'] - data['close'].shift(1)) * close_open_range * 
                    ((data['high'] - data['open']) - (data['open'] - data['low'])) / high_low_range_adj -
                    (data['close'].shift(1) - data['open']) * (data['open'] - data['close']) * 
                    ((data['open'] - data['low']) - (data['high'] - data['open'])) / high_low_range_adj)
    
    # Flow Asymmetry
    flow_asymmetry = (data['amount'] * (data['close'] - data['low']) / high_low_range_adj * 
                     ((data['close'] - data['low']) - (data['high'] - data['close'])) / high_low_range_adj -
                     data['amount'] * (data['high'] - data['close']) / high_low_range_adj * 
                     ((data['high'] - data['close']) - (data['close'] - data['low'])) / high_low_range_adj)
    
    # Flow Alignment (computed but not used in final composite)
    flow_alignment = (np.sign(flow_asymmetry) * np.sign(pressure_asymmetry) * 
                     np.sign(volume_asymmetry_divergence))
    
    # Composite Alpha Components
    volatility_driven_alpha = volatility_asymmetry_ratio * volatility_asymmetry_persistence
    pressure_driven_alpha = flow_asymmetry * pressure_asymmetry_persistence
    volume_driven_alpha = volume_asymmetry_divergence * volume_asymmetry_timing
    
    # Final Composite Alpha
    composite_alpha = volatility_driven_alpha + pressure_driven_alpha + volume_driven_alpha
    
    return composite_alpha
