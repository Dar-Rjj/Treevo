import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Momentum-Volume Divergence Asymmetry factor
    Combines momentum fractal analysis with asymmetric volume-price divergence patterns
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required minimum data length
    min_periods = 21
    
    for i in range(min_periods, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # 1. Hierarchical Momentum Fractal Analysis
        # Multi-Timeframe Momentum Fractals
        if i >= 2:
            mom_3d = current_data['close'].iloc[i] - current_data['close'].iloc[i-2]
        else:
            mom_3d = 0
            
        if i >= 7:
            mom_8d = current_data['close'].iloc[i] - current_data['close'].iloc[i-7]
        else:
            mom_8d = 0
            
        if i >= 20:
            mom_21d = current_data['close'].iloc[i] - current_data['close'].iloc[i-20]
        else:
            mom_21d = 0
        
        # Momentum Directional Consistency
        mom_signs = np.array([np.sign(mom_3d), np.sign(mom_8d), np.sign(mom_21d)])
        momentum_consistency = np.sum(mom_signs == mom_signs[0]) / len(mom_signs) if len(mom_signs) > 0 else 0
        
        # Momentum Scaling Discontinuities
        mom_magnitudes = np.array([abs(mom_3d), abs(mom_8d), abs(mom_21d)])
        if len(mom_magnitudes[mom_magnitudes > 0]) > 0:
            mom_scaling_ratio = np.std(mom_magnitudes[mom_magnitudes > 0]) / np.mean(mom_magnitudes[mom_magnitudes > 0])
        else:
            mom_scaling_ratio = 0
        
        # Momentum Regime Transition Signals
        momentum_regime = momentum_consistency * (1 - min(mom_scaling_ratio, 1))
        
        # 2. Volume-Price Divergence Asymmetry
        current_row = current_data.iloc[i]
        high_low_range = current_row['high'] - current_row['low']
        
        if high_low_range > 0:
            # Bullish Divergence Volume
            bullish_div_volume = current_row['volume'] * (current_row['close'] - current_row['low']) / high_low_range
            
            # Bearish Divergence Volume  
            bearish_div_volume = current_row['volume'] * (current_row['high'] - current_row['close']) / high_low_range
            
            # Divergence Asymmetry Ratio
            if bearish_div_volume > 0:
                divergence_asymmetry = (bullish_div_volume - bearish_div_volume) / (bullish_div_volume + bearish_div_volume)
            else:
                divergence_asymmetry = np.sign(bullish_div_volume)
        else:
            bullish_div_volume = 0
            bearish_div_volume = 0
            divergence_asymmetry = 0
        
        # Volume-Weighted Divergence Intensity
        volume_intensity = current_row['volume'] / current_data['volume'].iloc[max(0, i-4):i+1].mean() if i >= 4 else 1
        
        # Price-Range Divergence Persistence
        if i >= 4:
            recent_ranges = []
            for j in range(max(0, i-4), i+1):
                range_j = current_data['high'].iloc[j] - current_data['low'].iloc[j]
                if range_j > 0:
                    recent_ranges.append(range_j)
            range_persistence = np.std(recent_ranges) / np.mean(recent_ranges) if recent_ranges else 1
        else:
            range_persistence = 1
        
        # Divergence Acceleration Patterns
        if i >= 2:
            prev_asymmetry = 0
            count = 0
            for j in range(max(0, i-2), i):
                range_j = current_data['high'].iloc[j] - current_data['low'].iloc[j]
                if range_j > 0:
                    bull_j = current_data['volume'].iloc[j] * (current_data['close'].iloc[j] - current_data['low'].iloc[j]) / range_j
                    bear_j = current_data['volume'].iloc[j] * (current_data['high'].iloc[j] - current_data['close'].iloc[j]) / range_j
                    if bear_j > 0:
                        prev_asymmetry += (bull_j - bear_j) / (bull_j + bear_j)
                        count += 1
            if count > 0:
                divergence_acceleration = divergence_asymmetry - (prev_asymmetry / count)
            else:
                divergence_acceleration = divergence_asymmetry
        else:
            divergence_acceleration = divergence_asymmetry
        
        # Asymmetric Divergence Quality Score
        divergence_quality = (abs(divergence_asymmetry) * volume_intensity * 
                            (1 - min(range_persistence, 1)) * (1 + divergence_acceleration))
        
        # 3. Construct Momentum Divergence Alpha
        # Weight Divergence Signals by Momentum Fractal Strength
        momentum_weighted_divergence = divergence_quality * momentum_regime
        
        # Adjust Factor Sensitivity Based on Momentum Scaling
        momentum_scaling_factor = 1 + min(mom_scaling_ratio, 0.5)
        
        # Apply Momentum-Adaptive Smoothing
        if i >= 4:
            recent_values = []
            for j in range(max(0, i-4), i):
                if not pd.isna(result.iloc[j]):
                    recent_values.append(result.iloc[j])
            if recent_values:
                momentum_smoothing = np.mean(recent_values)
            else:
                momentum_smoothing = 0
        else:
            momentum_smoothing = 0
        
        # Calculate Fractal Range-Divergence Alignment
        range_divergence_alignment = (1 - min(range_persistence, 1)) * momentum_consistency
        
        # Final Alpha Construction
        alpha_value = (momentum_weighted_divergence * momentum_scaling_factor + 
                      momentum_smoothing * 0.3 + 
                      range_divergence_alignment * divergence_quality)
        
        result.iloc[i] = alpha_value
    
    # Fill initial values
    result = result.fillna(0)
    
    return result
