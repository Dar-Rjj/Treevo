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
        if i < 10:  # Need enough data for calculations
            alpha.iloc[i] = 0
            continue
            
        current = df.iloc[i]
        prev1 = df.iloc[i-1] if i >= 1 else None
        prev2 = df.iloc[i-2] if i >= 2 else None
        
        # Momentum-Entropy Components
        fractal_momentum = (current['close'] - current['open']) / (current['high'] - current['low'] + 1e-8)
        
        gap_magnitude = abs(current['close'] - current['open'])
        gap_ratio = gap_magnitude / (current['high'] - current['low'] + 1e-8)
        gap_entropy = -gap_magnitude * np.log(gap_ratio + 1e-8)
        
        momentum_entropy = fractal_momentum * gap_entropy
        
        # Volume-Compression Dynamics
        if prev1 is not None:
            compression_intensity = (current['high'] - current['low']) / (prev1['high'] - prev1['low'] + 1e-8)
            volume_ratio = current['volume'] / (prev1['volume'] + 1e-8)
            volume_entropy = -current['volume'] * np.log(volume_ratio + 1e-8)
            entangled_compression = compression_intensity * volume_entropy
        else:
            entangled_compression = 0
        
        # Pressure Dynamics
        upward_pressure = (current['close'] - current['open']) / (current['high'] - current['low'] + 1e-8)
        lower_pressure = (current['close'] - current['low']) / (current['high'] - current['low'] + 1e-8)
        pressure_divergence = upward_pressure - lower_pressure
        
        # Memory Propagation
        if prev1 is not None and prev2 is not None:
            prev2_fractal_momentum = (prev2['close'] - prev2['open']) / (prev2['high'] - prev2['low'] + 1e-8)
            momentum_memory = fractal_momentum / (prev2_fractal_momentum + 1e-8)
            volume_memory = current['volume'] / (prev1['volume'] + 1e-8)
            memory_factor = momentum_memory * volume_memory
        else:
            memory_factor = 1
        
        # Breakout Dynamics
        window_data = df.iloc[i-9:i+1]  # Last 10 days including current
        min_low_10 = window_data['low'].min()
        max_high_10 = window_data['high'].max()
        range_position = (current['close'] - min_low_10) / (max_high_10 - min_low_10 + 1e-8)
        
        # Cross-Scale Enhancement
        micro_efficiency = abs(current['close'] - current['open']) / (current['high'] - current['low'] + 1e-8)
        
        # Macro Efficiency (5-day)
        if i >= 5:
            window_5 = df.iloc[i-4:i+1]  # Last 5 days including current
            price_change_5 = abs(current['close'] - window_5.iloc[0]['close'])
            range_sum_5 = (window_5['high'] - window_5['low']).sum()
            macro_efficiency = price_change_5 / (range_sum_5 + 1e-8)
            cross_scale_factor = (macro_efficiency - micro_efficiency) * gap_entropy
        else:
            cross_scale_factor = 0
        
        # Alpha Synthesis
        core_signal = momentum_entropy * entangled_compression * pressure_divergence
        memory_enhanced = core_signal * memory_factor
        range_adjusted = memory_enhanced * range_position
        final_alpha = range_adjusted * (1 + cross_scale_factor)
        
        alpha.iloc[i] = final_alpha
    
    return alpha
