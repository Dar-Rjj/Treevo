import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Fractal Pressure with Volume-Elasticity Alignment factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    factor_values = pd.Series(index=df.index, dtype=float)
    
    for i in range(15, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Multi-Timeframe Fractal Analysis
        # 3-day fractal pressure efficiency
        if i >= 3:
            high_3d = current_data['high'].iloc[i-2:i+1].max()
            low_3d = current_data['low'].iloc[i-2:i+1].min()
            vol_sum_3d = current_data['volume'].iloc[i-2:i+1].sum()
            net_pressure_3d = ((current_data['close'].iloc[i] - current_data['close'].iloc[i-3]) / 
                             (high_3d - low_3d + 1e-8)) * vol_sum_3d
            fractal_efficiency_3d = net_pressure_3d / (vol_sum_3d + 1e-8)
        else:
            fractal_efficiency_3d = 0
        
        # 8-day fractal pressure efficiency
        if i >= 8:
            high_8d = current_data['high'].iloc[i-7:i+1].max()
            low_8d = current_data['low'].iloc[i-7:i+1].min()
            vol_sum_8d = current_data['volume'].iloc[i-7:i+1].sum()
            net_pressure_8d = ((current_data['close'].iloc[i] - current_data['close'].iloc[i-8]) / 
                             (high_8d - low_8d + 1e-8)) * vol_sum_8d
            fractal_efficiency_8d = net_pressure_8d / (vol_sum_8d + 1e-8)
        else:
            fractal_efficiency_8d = 0
        
        # 15-day fractal pressure efficiency
        if i >= 15:
            high_15d = current_data['high'].iloc[i-14:i+1].max()
            low_15d = current_data['low'].iloc[i-14:i+1].min()
            vol_sum_15d = current_data['volume'].iloc[i-14:i+1].sum()
            net_pressure_15d = ((current_data['close'].iloc[i] - current_data['close'].iloc[i-15]) / 
                              (high_15d - low_15d + 1e-8)) * vol_sum_15d
            fractal_efficiency_15d = net_pressure_15d / (vol_sum_15d + 1e-8)
        else:
            fractal_efficiency_15d = 0
        
        # Fractal Pressure Acceleration Analysis
        short_medium_acc = fractal_efficiency_3d - fractal_efficiency_8d
        medium_long_acc = fractal_efficiency_8d - fractal_efficiency_15d
        short_long_acc = fractal_efficiency_3d - fractal_efficiency_15d
        
        # Acceleration consistency assessment
        acc_consistency = (np.sign(short_medium_acc) + np.sign(medium_long_acc) + np.sign(short_long_acc)) / 3.0
        
        # Volume-Elasticity Pattern Recognition
        # Volume elasticity analysis
        if i >= 1:
            vol_elasticity = (current_data['volume'].iloc[i] / (current_data['volume'].iloc[i-1] + 1e-8))
        else:
            vol_elasticity = 1
        
        # Multi-period elasticity
        if i >= 3:
            price_change_3d = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-3]) / (current_data['close'].iloc[i-3] + 1e-8)
            range_3d = (high_3d - low_3d) / (current_data['close'].iloc[i] + 1e-8)
            multi_period_elasticity = price_change_3d / (range_3d + 1e-8)
        else:
            multi_period_elasticity = 0
        
        # Pressure accumulation mechanics
        high_current = current_data['high'].iloc[i]
        low_current = current_data['low'].iloc[i]
        close_current = current_data['close'].iloc[i]
        vol_current = current_data['volume'].iloc[i]
        
        intraday_pressure = ((close_current - low_current) / (high_current - low_current + 1e-8)) * vol_current
        selling_pressure = ((high_current - close_current) / (high_current - low_current + 1e-8)) * vol_current
        net_pressure_current = intraday_pressure - selling_pressure
        
        # Net pressure accumulation (4-day)
        if i >= 4:
            net_pressure_accum = 0
            for j in range(4):
                high_j = current_data['high'].iloc[i-j]
                low_j = current_data['low'].iloc[i-j]
                close_j = current_data['close'].iloc[i-j]
                vol_j = current_data['volume'].iloc[i-j]
                
                intraday_pressure_j = ((close_j - low_j) / (high_j - low_j + 1e-8)) * vol_j
                selling_pressure_j = ((high_j - close_j) / (high_j - low_j + 1e-8)) * vol_j
                net_pressure_j = intraday_pressure_j - selling_pressure_j
                net_pressure_accum += net_pressure_j
            
            pressure_gradient = net_pressure_current / (net_pressure_accum + 1e-8)
        else:
            pressure_gradient = 0
        
        # Volume-elasticity pattern analysis
        if i >= 5:
            vol_momentum = (current_data['volume'].iloc[i] / (current_data['volume'].iloc[i-5] + 1e-8) - 1)
        else:
            vol_momentum = 0
        
        volume_elasticity_efficiency = vol_elasticity * multi_period_elasticity
        
        # Pattern strength assessment
        pattern_strength = 0
        if multi_period_elasticity > 0.5 and pressure_gradient > 0:
            pattern_strength = 1
        elif multi_period_elasticity < 0.3 and pressure_gradient < 0:
            pattern_strength = -1
        
        # Fractal Pressure-Volume Elasticity Integration
        # Volume-fractal pressure comparison
        volume_fractal_alignment = 0
        if vol_elasticity > 1 and fractal_efficiency_3d > 0:
            volume_fractal_alignment = 1
        elif vol_elasticity < 1 and fractal_efficiency_3d < 0:
            volume_fractal_alignment = -1
        
        # Elasticity-adaptive pattern weighting
        elasticity_weight = multi_period_elasticity if multi_period_elasticity > 0.3 else 1.0
        
        # Multi-timeframe pattern consistency
        timeframe_consistency = 0
        if (fractal_efficiency_3d > 0 and fractal_efficiency_8d > 0 and fractal_efficiency_15d > 0):
            timeframe_consistency = 1
        elif (fractal_efficiency_3d < 0 and fractal_efficiency_8d < 0 and fractal_efficiency_15d < 0):
            timeframe_consistency = -1
        
        # Composite Factor Generation
        # Base signal from fractal acceleration
        base_signal = (short_medium_acc * 0.4 + medium_long_acc * 0.3 + short_long_acc * 0.3)
        
        # Apply elasticity-adaptive weighting
        weighted_signal = base_signal * elasticity_weight
        
        # Volume-elasticity validation
        validated_signal = weighted_signal * (1 + 0.5 * volume_fractal_alignment)
        
        # Pattern strength enhancement
        enhanced_signal = validated_signal * (1 + 0.3 * pattern_strength)
        
        # Timeframe consistency boost
        final_signal = enhanced_signal * (1 + 0.2 * timeframe_consistency)
        
        # Acceleration consistency refinement
        refined_signal = final_signal * (1 + 0.1 * acc_consistency)
        
        factor_values.iloc[i] = refined_signal
    
    # Fill initial NaN values with 0
    factor_values = factor_values.fillna(0)
    
    return factor_values
