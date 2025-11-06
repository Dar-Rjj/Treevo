import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(5, len(df)):
        current_data = df.iloc[:i+1]
        
        # Gap Fractal
        if i >= 4:
            overnight_gap = (current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) / \
                           (current_data['high'].iloc[i-3] - current_data['low'].iloc[i-3])
            
            gap_persistence = (current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) / \
                             (current_data['open'].iloc[i-3] - current_data['close'].iloc[i-4])
        else:
            overnight_gap = 0
            gap_persistence = 0
        
        # Shadow Analysis
        if i >= 2:
            max_close_open = max(current_data['close'].iloc[i], current_data['open'].iloc[i])
            min_close_open = min(current_data['close'].iloc[i], current_data['open'].iloc[i])
            
            upper_shadow = (current_data['high'].iloc[i] - max_close_open) / \
                          (current_data['high'].iloc[i-2] - current_data['low'].iloc[i-2])
            
            lower_shadow = (min_close_open - current_data['low'].iloc[i]) / \
                          (current_data['high'].iloc[i-2] - current_data['low'].iloc[i-2])
            
            net_shadow_bias = lower_shadow - upper_shadow
        else:
            net_shadow_bias = 0
        
        # Volume Dynamics
        volume_intensity = current_data['volume'].iloc[i] / \
                          (abs(current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) * \
                           (current_data['high'].iloc[i] - current_data['low'].iloc[i]))
        
        # Volume timing - using rolling window of last 5 days
        if i >= 4:
            window_data = current_data.iloc[i-4:i+1]
            volume_argmax = window_data['volume'].values.argmax()
            price_change_argmax = abs(window_data['close'].values - window_data['open'].values).argmax()
            
            volume_timing = (volume_argmax - price_change_argmax) / \
                           (current_data['high'].iloc[i-2] - current_data['low'].iloc[i-2])
        else:
            volume_timing = 0
        
        # Market Regime
        if i >= 5:
            volatility_ratio = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) / \
                              (current_data['high'].iloc[i-5] - current_data['low'].iloc[i-5])
        else:
            volatility_ratio = 1
        
        hl_midpoint = (current_data['high'].iloc[i] + current_data['low'].iloc[i]) / 2
        spread_measure = 2 * abs(current_data['close'].iloc[i] - hl_midpoint) / hl_midpoint
        
        # Alpha Composition
        momentum_core = overnight_gap * net_shadow_bias
        volume_factor = volume_intensity * volume_timing
        
        # Final alpha calculation
        volatility_adjustment = 1 + abs(volatility_ratio - 1) * spread_measure
        final_alpha = (momentum_core + volume_factor) * volatility_adjustment
        
        result.iloc[i] = final_alpha
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
