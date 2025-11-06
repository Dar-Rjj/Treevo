import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 20:  # Need at least 20 days for calculations
            result.iloc[i] = 0
            continue
            
        current = df.iloc[i]
        # Get historical data using only past information
        past_data = df.iloc[:i+1]
        
        # Gap Momentum Divergence
        # Micro Gap Momentum
        micro_gap_momentum = (current['close'] - current['open']) / (current['high'] - current['low'] + 1e-8)
        
        # Macro Gap Momentum
        if i >= 5:
            close_t_minus_5 = df.iloc[i-5]['close']
            high_low_sum = sum(df.iloc[j]['high'] - df.iloc[j]['low'] for j in range(i-4, i+1))
            macro_gap_momentum = (current['close'] - close_t_minus_5) / (high_low_sum + 1e-8)
        else:
            macro_gap_momentum = 0
        
        # Gap Momentum Divergence
        if i >= 1:
            close_t_minus_1 = df.iloc[i-1]['close']
            gap_momentum_divergence = (macro_gap_momentum - micro_gap_momentum) * np.sign(current['close'] - close_t_minus_1)
        else:
            gap_momentum_divergence = 0
        
        # Volume-Gap Coherence
        gap_efficiency = abs(current['close'] - current['open']) / (current['high'] - current['low'] + 1e-8)
        
        if i >= 1:
            volume_t_minus_1 = df.iloc[i-1]['volume']
            coherence_signal = np.sign(current['close'] - current['open']) * np.sign(current['volume'] - volume_t_minus_1) * gap_efficiency
        else:
            coherence_signal = 0
        
        # Pressure Asymmetry
        upward_pressure = (current['close'] - current['open']) / (current['high'] - current['low'] + 1e-8)
        lower_pressure = (current['close'] - current['low']) / (current['high'] - current['low'] + 1e-8)
        pressure_divergence = upward_pressure - lower_pressure
        
        # Volume Confirmation
        # Volume Fractal Dimension
        if i >= 20:
            volume_sum_5 = sum(df.iloc[j]['volume'] for j in range(i-4, i+1))
            volume_sum_20 = sum(df.iloc[j]['volume'] for j in range(i-19, i+1))
            volume_fractal_dimension = np.log(volume_sum_5 + 1e-8) / np.log(volume_sum_20 + 1e-8)
        else:
            volume_fractal_dimension = 1
        
        # Volume Momentum
        if i >= 10:
            volume_t_minus_5 = df.iloc[i-5]['volume']
            volume_t_minus_10 = df.iloc[i-10]['volume']
            if abs(volume_t_minus_5 - volume_t_minus_10) > 1e-8:
                volume_momentum = (current['volume'] - volume_t_minus_5) / (volume_t_minus_5 - volume_t_minus_10 + 1e-8)
            else:
                volume_momentum = 0
        else:
            volume_momentum = 0
        
        volume_confirmation = volume_fractal_dimension * volume_momentum
        
        # Core Signal
        base_alpha = gap_momentum_divergence * coherence_signal * pressure_divergence
        volume_enhanced_alpha = base_alpha * (1 + volume_confirmation)
        
        # Regime Adjustment
        efficiency_regime = 1 if gap_efficiency > 0.6 else 0
        volume_regime = 1 if volume_confirmation > 0 else 0
        final_alpha = volume_enhanced_alpha * (1 + 0.3 * efficiency_regime + 0.2 * volume_regime)
        
        result.iloc[i] = final_alpha
    
    return result
