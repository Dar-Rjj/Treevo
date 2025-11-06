import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize all required columns
    data['volatility_asymmetry_ratio'] = 0.0
    data['volatility_asymmetry_acceleration'] = 0.0
    data['volatility_asymmetry_persistence'] = 0.0
    data['pressure_asymmetry'] = 0.0
    data['pressure_asymmetry_shift'] = 0.0
    data['pressure_asymmetry_persistence'] = 0.0
    data['volume_asymmetry_divergence'] = 0.0
    data['volume_asymmetry_acceleration'] = 0.0
    data['volume_asymmetry_timing'] = 0.0
    data['gap_asymmetry'] = 0.0
    data['gap_efficiency'] = 0.0
    data['gap_efficiency_momentum'] = 0.0
    data['flow_asymmetry'] = 0.0
    
    # Calculate daily components
    for i in range(len(data)):
        if i < 2:  # Need at least 3 days for some calculations
            continue
            
        open_t = data['open'].iloc[i]
        high_t = data['high'].iloc[i]
        low_t = data['low'].iloc[i]
        close_t = data['close'].iloc[i]
        volume_t = data['volume'].iloc[i]
        amount_t = data['amount'].iloc[i]
        
        close_t_1 = data['close'].iloc[i-1]
        open_t_1 = data['open'].iloc[i-1]
        
        # Asymmetric Volatility Wave
        high_low_range = high_t - low_t
        if high_low_range == 0:
            high_low_range = 1e-8
            
        close_open_diff = close_t - open_t
        high_close_diff = high_t - close_t
        close_low_diff = close_t - low_t
        
        # Directional Volatility Asymmetry
        if close_t > open_t:
            up_day_vol_asymmetry = (high_low_range / (close_open_diff + 1e-8)) * ((close_low_diff - high_close_diff) / high_low_range)
        else:
            up_day_vol_asymmetry = 0.0
            
        if close_t < open_t:
            down_day_vol_asymmetry = (high_low_range / (open_t - close_t + 1e-8)) * ((high_close_diff - close_low_diff) / high_low_range)
        else:
            down_day_vol_asymmetry = 0.0
            
        vol_asymmetry_ratio = up_day_vol_asymmetry / (down_day_vol_asymmetry + 1e-8)
        data.loc[data.index[i], 'volatility_asymmetry_ratio'] = vol_asymmetry_ratio
        
        # Volatility Asymmetry Acceleration
        if i >= 1:
            vol_asymmetry_ratio_t_1 = data['volatility_asymmetry_ratio'].iloc[i-1]
            if vol_asymmetry_ratio_t_1 != 0:
                vol_asymmetry_acceleration = vol_asymmetry_ratio / vol_asymmetry_ratio_t_1 - 1
                data.loc[data.index[i], 'volatility_asymmetry_acceleration'] = vol_asymmetry_acceleration
        
        # Volatility Asymmetry Persistence
        if i >= 3:
            signs = []
            for j in range(i-2, i+1):
                if j >= 1:
                    acc_j = data['volatility_asymmetry_acceleration'].iloc[j]
                    acc_j_1 = data['volatility_asymmetry_acceleration'].iloc[j-1]
                    if acc_j != 0 and acc_j_1 != 0:
                        signs.append(np.sign(acc_j) == np.sign(acc_j_1))
            if len(signs) > 0:
                persistence = sum(signs) / len(signs)
                data.loc[data.index[i], 'volatility_asymmetry_persistence'] = persistence
        
        # Asymmetric Pressure Dynamics
        opening_pressure = (open_t - low_t) / high_low_range
        closing_pressure = (close_t - low_t) / high_low_range
        pressure_asymmetry = (close_low_diff - high_close_diff) / high_low_range
        data.loc[data.index[i], 'pressure_asymmetry'] = pressure_asymmetry
        
        # Pressure Asymmetry Shift (3-day average)
        if i >= 3:
            pressure_3day_avg = data['pressure_asymmetry'].iloc[i-3:i].mean()
            pressure_shift = pressure_asymmetry - pressure_3day_avg
            data.loc[data.index[i], 'pressure_asymmetry_shift'] = pressure_shift
        
        # Pressure Asymmetry Persistence
        if i >= 3:
            signs = []
            for j in range(i-2, i+1):
                if j >= 1:
                    pa_j = data['pressure_asymmetry'].iloc[j]
                    pa_j_1 = data['pressure_asymmetry'].iloc[j-1]
                    if pa_j != 0 and pa_j_1 != 0:
                        signs.append(np.sign(pa_j) == np.sign(pa_j_1))
            if len(signs) > 0:
                persistence = sum(signs) / len(signs)
                data.loc[data.index[i], 'pressure_asymmetry_persistence'] = persistence
        
        # Asymmetric Volume Wave
        high_vol_asymmetry = volume_t * high_low_range * ((close_low_diff - high_close_diff) / high_low_range)
        low_vol_asymmetry = volume_t * abs(close_open_diff) * ((high_close_diff - close_low_diff) / high_low_range)
        volume_asymmetry_divergence = high_vol_asymmetry - low_vol_asymmetry
        data.loc[data.index[i], 'volume_asymmetry_divergence'] = volume_asymmetry_divergence
        
        # Volume Asymmetry Acceleration
        if i >= 2:
            vad_t = volume_asymmetry_divergence
            vad_t_1 = data['volume_asymmetry_divergence'].iloc[i-1]
            vad_t_2 = data['volume_asymmetry_divergence'].iloc[i-2]
            
            if vad_t_1 != 0 and vad_t_2 != 0:
                acc_t = (vad_t / vad_t_1 - 1) if vad_t_1 != 0 else 0
                acc_t_1 = (vad_t_1 / vad_t_2 - 1) if vad_t_2 != 0 else 0
                volume_acceleration = (acc_t - acc_t_1) * np.sign(close_t - close_t_1)
                data.loc[data.index[i], 'volume_asymmetry_acceleration'] = volume_acceleration
        
        # Volume Asymmetry Timing
        if i >= 3:
            signs = []
            for j in range(i-2, i+1):
                if j >= 2:
                    vad_j = data['volume_asymmetry_divergence'].iloc[j]
                    vad_j_1 = data['volume_asymmetry_divergence'].iloc[j-1]
                    vad_j_2 = data['volume_asymmetry_divergence'].iloc[j-2]
                    
                    if vad_j_1 != 0 and vad_j_2 != 0:
                        acc_j = (vad_j / vad_j_1 - 1) if vad_j_1 != 0 else 0
                        acc_j_1 = (vad_j_1 / vad_j_2 - 1) if vad_j_2 != 0 else 0
                        signs.append(np.sign(acc_j) == np.sign(acc_j_1))
            if len(signs) > 0:
                timing = sum(signs) / len(signs)
                data.loc[data.index[i], 'volume_asymmetry_timing'] = timing
        
        # Asymmetric Gap Wave
        gap_open_close_prev = open_t - close_t_1
        
        if gap_open_close_prev > 0:  # Up gap
            up_gap_momentum = gap_open_close_prev * (close_t - open_t) * ((high_t - open_t) - (open_t - low_t)) / high_low_range
            down_gap_momentum = 0.0
        elif gap_open_close_prev < 0:  # Down gap
            up_gap_momentum = 0.0
            down_gap_momentum = (close_t_1 - open_t) * (open_t - close_t) * ((open_t - low_t) - (high_t - open_t)) / high_low_range
        else:
            up_gap_momentum = 0.0
            down_gap_momentum = 0.0
            
        gap_asymmetry = up_gap_momentum - down_gap_momentum
        data.loc[data.index[i], 'gap_asymmetry'] = gap_asymmetry
        
        # Gap Efficiency
        gap_efficiency = abs(gap_asymmetry) / high_low_range if high_low_range != 0 else 0.0
        data.loc[data.index[i], 'gap_efficiency'] = gap_efficiency
        
        # Gap Efficiency Momentum
        if i >= 1:
            gap_eff_t_1 = data['gap_efficiency'].iloc[i-1]
            if gap_eff_t_1 != 0:
                gap_eff_momentum = gap_efficiency / gap_eff_t_1 - 1
                data.loc[data.index[i], 'gap_efficiency_momentum'] = gap_eff_momentum
        
        # Asymmetric Flow Wave
        if close_t > open_t:
            buy_flow_asymmetry = amount_t * (close_low_diff / high_low_range) * ((close_low_diff - high_close_diff) / high_low_range)
            sell_flow_asymmetry = 0.0
        elif close_t < open_t:
            buy_flow_asymmetry = 0.0
            sell_flow_asymmetry = amount_t * (high_close_diff / high_low_range) * ((high_close_diff - close_low_diff) / high_low_range)
        else:
            buy_flow_asymmetry = 0.0
            sell_flow_asymmetry = 0.0
            
        flow_asymmetry = buy_flow_asymmetry - sell_flow_asymmetry
        data.loc[data.index[i], 'flow_asymmetry'] = flow_asymmetry
    
    # Multi-Scale Wave Synthesis
    data['volatility_wave_factor'] = data['volatility_asymmetry_ratio'] * data['pressure_asymmetry_shift']
    data['volume_wave_factor'] = data['volume_asymmetry_divergence'] * data['volume_asymmetry_timing']
    data['gap_wave_factor'] = data['gap_asymmetry'] * data['gap_efficiency_momentum']
    
    # Persistence-Enhanced Asymmetry
    data['volatility_persistence_weight'] = data['volatility_wave_factor'] * data['volatility_asymmetry_persistence']
    data['pressure_persistence_weight'] = data['flow_asymmetry'] * data['pressure_asymmetry_persistence']
    data['volume_timing_weight'] = data['volume_wave_factor'] * data['volume_asymmetry_timing']
    
    # Composite Asymmetric Wave Alpha
    data['volatility_driven_alpha'] = data['volatility_persistence_weight'] * 1.3
    data['pressure_driven_alpha'] = data['pressure_persistence_weight'] * 1.4
    data['volume_driven_alpha'] = data['volume_timing_weight'] * 1.2
    
    # Final Alpha Construction
    data['composite_alpha'] = data['volatility_driven_alpha'] + data['pressure_driven_alpha'] + data['volume_driven_alpha']
    
    return data['composite_alpha']
