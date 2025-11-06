import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required intermediate features
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Calculate price changes
    close_change = close.diff()
    
    # Asymmetric Pressure Analysis
    up_tick_pressure = pd.Series(0.0, index=df.index)
    down_tick_pressure = pd.Series(0.0, index=df.index)
    
    for i in range(5):  # i=0 to 4
        shifted_close = close.shift(i)
        shifted_volume = volume.shift(i)
        prev_close = close.shift(i+1)
        
        # Identify up-tick and down-tick periods
        up_mask = shifted_close > prev_close
        down_mask = shifted_close < prev_close
        
        # Calculate rolling sums
        up_tick_pressure += up_mask.astype(float) * shifted_volume
        down_tick_pressure += down_mask.astype(float) * shifted_volume
    
    # Pressure Asymmetry
    pressure_asymmetry = (up_tick_pressure - down_tick_pressure) / (up_tick_pressure + down_tick_pressure + 1e-8)
    
    # Entropy Dynamics
    volume_entropy = pd.Series(0.0, index=df.index)
    
    for i in range(len(df)):
        if i >= 4:  # Need at least 5 periods for calculation
            volume_window = volume.iloc[i-4:i+1]  # t-4 to t
            total_volume = volume_window.sum()
            
            if total_volume > 0:
                probabilities = volume_window / total_volume
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
                volume_entropy.iloc[i] = entropy
    
    # Entropy Change
    entropy_change = volume_entropy.diff()
    
    # Price Impact Analysis
    upside_impact = (high - close) / (close + 1e-8)
    downside_impact = (close - low) / (close + 1e-8)
    impact_asymmetry = upside_impact - downside_impact
    
    # Efficiency Measurement
    high_low_range_sum = pd.Series(0.0, index=df.index)
    
    for i in range(5):  # i=0 to 4
        high_low_range_sum += (high.shift(i) - low.shift(i))
    
    upside_efficiency = (high - close.shift(1)) / (high_low_range_sum + 1e-8)
    downside_efficiency = (close.shift(1) - low) / (high_low_range_sum + 1e-8)
    efficiency_asymmetry = upside_efficiency - downside_efficiency
    
    # Core Alpha Factors
    entropy_pressure_factor = pressure_asymmetry * entropy_change
    impact_entropy_factor = impact_asymmetry * volume_entropy
    efficiency_pressure_factor = efficiency_asymmetry * pressure_asymmetry
    
    # Adaptive Alpha Synthesis
    high_entropy_alpha = 0.6 * entropy_pressure_factor + 0.4 * impact_entropy_factor
    low_entropy_alpha = 0.7 * efficiency_pressure_factor + 0.3 * impact_entropy_factor
    transition_alpha = 0.5 * entropy_pressure_factor + 0.5 * efficiency_pressure_factor
    
    # Final alpha factor - blend based on entropy level
    # Use high entropy alpha when entropy is high, low entropy alpha when entropy is low
    entropy_threshold = volume_entropy.median()
    
    final_alpha = pd.Series(0.0, index=df.index)
    high_entropy_mask = volume_entropy > entropy_threshold
    low_entropy_mask = volume_entropy <= entropy_threshold
    
    final_alpha[high_entropy_mask] = high_entropy_alpha[high_entropy_mask]
    final_alpha[low_entropy_mask] = low_entropy_alpha[low_entropy_mask]
    
    # Use transition alpha for boundary cases
    boundary_mask = (volume_entropy >= entropy_threshold * 0.95) & (volume_entropy <= entropy_threshold * 1.05)
    final_alpha[boundary_mask] = transition_alpha[boundary_mask]
    
    return final_alpha
