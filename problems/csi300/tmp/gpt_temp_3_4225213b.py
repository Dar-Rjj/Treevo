import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required rolling windows
    for i in range(len(df)):
        if i < 21:  # Need at least 21 days of data
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]
        
        # Multi-Timeframe Efficiency Analysis
        eff_3d = calculate_efficiency(current_data, i, 3)
        eff_8d = calculate_efficiency(current_data, i, 8)
        eff_21d = calculate_efficiency(current_data, i, 21)
        
        # Multi-Timeframe Volatility Adjustment
        vol_3d = calculate_volatility(current_data, i, 3)
        vol_8d = calculate_volatility(current_data, i, 8)
        vol_21d = calculate_volatility(current_data, i, 21)
        
        # Volatility-Efficiency Synchronization
        vol_eff_3d = eff_3d / (vol_3d + 1e-8)
        vol_eff_8d = eff_8d / (vol_8d + 1e-8)
        vol_eff_21d = eff_21d / (vol_21d + 1e-8)
        
        # Multi-Timeframe Pressure Dynamics
        press_3d = calculate_pressure(current_data, i, 3)
        press_8d = calculate_pressure(current_data, i, 8)
        press_21d = calculate_pressure(current_data, i, 21)
        
        # Volume Efficiency Analysis
        vol_momentum = calculate_volume_momentum(current_data, i)
        vol_range_eff = calculate_volume_range_efficiency(current_data, i)
        vol_price_align = calculate_volume_price_alignment(current_data, i)
        
        # Pressure-Volume Integration
        press_vol_3d = press_3d * vol_momentum * vol_range_eff
        press_vol_8d = press_8d * vol_momentum * vol_range_eff
        press_vol_21d = press_21d * vol_momentum * vol_range_eff
        
        # Volatility-Adjusted Momentum Components
        mom_short = calculate_momentum(current_data, i, 5)
        mom_medium = calculate_momentum(current_data, i, 20)
        mom_accel = mom_short - mom_medium
        
        # Multi-Timeframe Breakout Analysis
        breakout_3d = calculate_breakout(current_data, i, 3)
        breakout_8d = calculate_breakout(current_data, i, 8)
        breakout_21d = calculate_breakout(current_data, i, 21)
        
        # Breakout Volume-Pressure Confirmation
        vol_breakout_ratio = calculate_volume_breakout_ratio(current_data, i)
        press_breakout_intensity = calculate_pressure_breakout_intensity(current_data, i)
        
        # Core Factor Construction
        vol_eff_alignment = (vol_eff_3d + vol_eff_8d + vol_eff_21d) / 3
        press_vol_confirmation = (press_vol_3d + press_vol_8d + press_vol_21d) / 3
        
        primary_factor = vol_eff_alignment * press_vol_confirmation
        momentum_enhanced = primary_factor * mom_accel
        
        breakout_efficiency = (breakout_3d + breakout_8d + breakout_21d) / 3
        final_alpha = momentum_enhanced * breakout_efficiency * vol_price_align
        
        # Apply non-linear transformation
        result.iloc[i] = np.tanh(final_alpha)
    
    return result

def calculate_efficiency(data, current_idx, period):
    if current_idx < period:
        return 0
    
    close_current = data['close'].iloc[current_idx]
    close_prev = data['close'].iloc[current_idx - period]
    price_change = close_current - close_prev
    
    volume_weighted_vol = 0
    for j in range(current_idx - period + 1, current_idx + 1):
        if j > 0:
            daily_vol = abs(data['close'].iloc[j] - data['close'].iloc[j-1])
            volume_weighted_vol += data['volume'].iloc[j] * daily_vol
    
    return price_change / (volume_weighted_vol + 1e-8)

def calculate_volatility(data, current_idx, period):
    if current_idx < period:
        return 0
    
    total_vol = 0
    for j in range(current_idx - period + 1, current_idx + 1):
        if j > 0:
            total_vol += abs(data['close'].iloc[j] - data['close'].iloc[j-1])
    
    high_max = data['high'].iloc[current_idx - period + 1:current_idx + 1].max()
    low_min = data['low'].iloc[current_idx - period + 1:current_idx + 1].min()
    price_range = high_max - low_min
    
    return total_vol / (price_range + 1e-8)

def calculate_pressure(data, current_idx, period):
    if current_idx < period:
        return 0
    
    close_current = data['close'].iloc[current_idx]
    low_min = data['low'].iloc[current_idx - period + 1:current_idx + 1].min()
    high_max = data['high'].iloc[current_idx - period + 1:current_idx + 1].max()
    
    return (close_current - low_min) / (high_max - low_min + 1e-8)

def calculate_volume_momentum(data, current_idx):
    if current_idx < 5:
        return 1
    
    current_volume = data['volume'].iloc[current_idx]
    avg_volume = data['volume'].iloc[current_idx - 5:current_idx].mean()
    
    return current_volume / (avg_volume + 1e-8)

def calculate_volume_range_efficiency(data, current_idx):
    current_volume = data['volume'].iloc[current_idx]
    price_range = data['high'].iloc[current_idx] - data['low'].iloc[current_idx]
    
    return current_volume / (price_range + 1e-8)

def calculate_volume_price_alignment(data, current_idx):
    if current_idx < 1:
        return 0
    
    price_change_sign = np.sign(data['close'].iloc[current_idx] - data['close'].iloc[current_idx - 1])
    volume_change_sign = np.sign(data['volume'].iloc[current_idx] - data['volume'].iloc[current_idx - 1])
    
    return price_change_sign * volume_change_sign

def calculate_momentum(data, current_idx, period):
    if current_idx < period:
        return 0
    
    return data['close'].iloc[current_idx] / data['close'].iloc[current_idx - period] - 1

def calculate_breakout(data, current_idx, period):
    if current_idx < period:
        return 0
    
    close_current = data['close'].iloc[current_idx]
    high_max = data['high'].iloc[current_idx - period + 1:current_idx + 1].max()
    
    return close_current / high_max

def calculate_volume_breakout_ratio(data, current_idx):
    if current_idx < 6:
        return 1
    
    current_volume = data['volume'].iloc[current_idx]
    avg_volume = data['volume'].iloc[current_idx - 5:current_idx].mean()
    
    return current_volume / (avg_volume + 1e-8)

def calculate_pressure_breakout_intensity(data, current_idx):
    close_current = data['close'].iloc[current_idx]
    high_current = data['high'].iloc[current_idx]
    low_current = data['low'].iloc[current_idx]
    
    return (close_current - low_current) / (high_current - low_current + 1e-8)
