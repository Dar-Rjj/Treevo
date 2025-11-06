import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 5:  # Need at least 5 days for some calculations
            alpha.iloc[i] = 0
            continue
            
        current = df.iloc[i]
        prev_day = df.iloc[i-1] if i >= 1 else None
        
        # Efficiency Momentum
        high_low_range = current['high'] - current['low']
        if high_low_range == 0:
            intraday_efficiency = 0
            gap_efficiency = 0
        else:
            intraday_efficiency = (current['close'] - current['open']) * current['volume'] / high_low_range
            if prev_day is not None:
                gap_efficiency = (current['open'] - prev_day['close']) * current['volume'] / high_low_range
            else:
                gap_efficiency = 0
        
        # Microstructure Pressure
        hl_midpoint = (current['high'] + current['low']) / 2
        close_open_sign = np.sign(current['close'] - current['open'])
        if close_open_sign == 0:
            close_open_sign = 1
            
        closing_pressure = (current['close'] - hl_midpoint) * close_open_sign * current['volume']
        
        if current['close'] == 0:
            effective_spread = 0
        else:
            effective_spread = 2 * abs(current['close'] - hl_midpoint) / current['close'] * close_open_sign
        
        # Volume Dynamics
        close_open_diff = abs(current['close'] - current['open'])
        if close_open_diff == 0:
            volume_per_movement = 0
        else:
            volume_per_movement = current['volume'] / close_open_diff
            
        if high_low_range == 0:
            volume_to_range = 0
        else:
            volume_to_range = current['volume'] / high_low_range
            
        # Volume Momentum
        if i >= 1:
            prev_close_open_diff = abs(df.iloc[i-1]['close'] - df.iloc[i-1]['open'])
            if prev_close_open_diff == 0:
                prev_volume_per_movement = 0
            else:
                prev_volume_per_movement = df.iloc[i-1]['volume'] / prev_close_open_diff
            volume_momentum = np.sign(volume_per_movement - prev_volume_per_movement)
        else:
            volume_momentum = 0
        
        # Convergence Patterns
        efficiency_pressure_alignment = intraday_efficiency * closing_pressure
        volume_efficiency_confirmation = volume_momentum * np.sign(intraday_efficiency) if intraday_efficiency != 0 else 0
        microstructure_convergence = effective_spread * gap_efficiency
        
        # Multi-Timeframe Analysis
        # Short-Term Trend (Efficiency_t / Efficiency_t-5)
        if i >= 5:
            prev_high_low_range_5 = df.iloc[i-5]['high'] - df.iloc[i-5]['low']
            if prev_high_low_range_5 == 0:
                prev_intraday_efficiency_5 = 0
            else:
                prev_intraday_efficiency_5 = (df.iloc[i-5]['close'] - df.iloc[i-5]['open']) * df.iloc[i-5]['volume'] / prev_high_low_range_5
            
            if prev_intraday_efficiency_5 == 0:
                short_term_trend = 1
            else:
                short_term_trend = intraday_efficiency / prev_intraday_efficiency_5
        else:
            short_term_trend = 1
        
        # Pressure Consistency (count sign consistent over 5 days)
        pressure_signs = []
        for j in range(max(0, i-4), i+1):
            day_j = df.iloc[j]
            hl_mid_j = (day_j['high'] + day_j['low']) / 2
            co_sign_j = np.sign(day_j['close'] - day_j['open'])
            if co_sign_j == 0:
                co_sign_j = 1
            cp_j = (day_j['close'] - hl_mid_j) * co_sign_j * day_j['volume']
            pressure_signs.append(np.sign(cp_j))
        
        if len(pressure_signs) >= 2:
            pressure_consistency = sum(1 for k in range(1, len(pressure_signs)) 
                                   if pressure_signs[k] == pressure_signs[0] and pressure_signs[k] != 0)
        else:
            pressure_consistency = 0
        
        # Volume Stability
        if i >= 5:
            prev_high_low_range_5_vr = df.iloc[i-5]['high'] - df.iloc[i-5]['low']
            if prev_high_low_range_5_vr == 0:
                prev_volume_to_range_5 = 0
            else:
                prev_volume_to_range_5 = df.iloc[i-5]['volume'] / prev_high_low_range_5_vr
            
            if prev_volume_to_range_5 == 0:
                volume_stability = 1
            else:
                volume_stability = volume_to_range / prev_volume_to_range_5
        else:
            volume_stability = 1
        
        # Final Alpha Signals
        bullish_conditions = (
            efficiency_pressure_alignment > 0 and
            volume_efficiency_confirmation > 0 and
            short_term_trend > 1
        )
        
        bearish_conditions = (
            efficiency_pressure_alignment < 0 and
            volume_efficiency_confirmation < 0 and
            short_term_trend < 1
        )
        
        # Calculate final alpha value
        if bullish_conditions:
            alpha_value = (efficiency_pressure_alignment * 0.4 + 
                          volume_efficiency_confirmation * 0.3 + 
                          (short_term_trend - 1) * 0.3)
        elif bearish_conditions:
            alpha_value = (efficiency_pressure_alignment * 0.4 + 
                          volume_efficiency_confirmation * 0.3 + 
                          (short_term_trend - 1) * 0.3)
        else:
            # Neutral signal based on convergence patterns
            alpha_value = (efficiency_pressure_alignment * 0.5 + 
                          microstructure_convergence * 0.3 + 
                          pressure_consistency * 0.2)
        
        alpha.iloc[i] = alpha_value
    
    return alpha
