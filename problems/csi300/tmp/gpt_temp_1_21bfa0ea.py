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
        if i < 9:  # Need at least 9 days for medium-term calculations
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[i]
        prev_data = df.iloc[i-1] if i >= 1 else None
        prev2_data = df.iloc[i-2] if i >= 2 else None
        
        # Bidirectional Order Flow Imbalance
        # Micro-level pressure
        if current_data['high'] != current_data['low']:
            micro_pressure = ((current_data['close'] - current_data['open']) * 
                            current_data['volume'] / (current_data['high'] - current_data['low']))
        else:
            micro_pressure = 0
            
        # Intraday absorption
        if current_data['high'] != current_data['low']:
            intraday_absorption = (((current_data['high'] - current_data['close']) - 
                                  (current_data['close'] - current_data['low'])) * 
                                 current_data['volume'] / (current_data['high'] - current_data['low']))
        else:
            intraday_absorption = 0
            
        flow_imbalance = micro_pressure + intraday_absorption
        
        # Multi-Timeframe Price Rejection
        # Short-term rejection
        if (prev_data is not None and 
            current_data['high'] != current_data['low'] and 
            prev_data['high'] != prev_data['low']):
            short_term_rej = ((current_data['close'] - current_data['low']) / 
                            (current_data['high'] - current_data['low']) - 
                            (prev_data['close'] - prev_data['low']) / 
                            (prev_data['high'] - prev_data['low']))
        else:
            short_term_rej = 0
            
        # Medium-term rejection
        # Current window (t-4 to t)
        current_window = df.iloc[max(0, i-4):i+1]
        current_min_low = current_window['low'].min()
        current_max_high = current_window['high'].max()
        
        # Previous window (t-9 to t-5)
        prev_window = df.iloc[max(0, i-9):max(0, i-4)]
        prev_min_low = prev_window['low'].min() if len(prev_window) > 0 else 0
        prev_max_high = prev_window['high'].max() if len(prev_window) > 0 else 0
        
        if (current_max_high != current_min_low and 
            prev_max_high != prev_min_low and 
            len(prev_window) > 0):
            medium_term_rej = ((current_data['close'] - current_min_low) / 
                             (current_max_high - current_min_low) - 
                             (df.iloc[i-5]['close'] - prev_min_low) / 
                             (prev_max_high - prev_min_low))
        else:
            medium_term_rej = 0
            
        rejection_momentum = short_term_rej * medium_term_rej
        
        # Volume-Value Dislocation
        # Value density
        if current_data['volume'] > 0:
            value_density = current_data['amount'] / current_data['volume']
        else:
            value_density = 0
            
        # Volume acceleration
        if (prev_data is not None and prev2_data is not None and 
            prev_data['volume'] > 0 and prev2_data['volume'] > 0):
            volume_acceleration = (current_data['volume'] / prev_data['volume'] - 
                                 prev_data['volume'] / prev2_data['volume'])
        else:
            volume_acceleration = 0
            
        dislocation_factor = value_density * volume_acceleration
        
        # Adaptive Microstructure Synthesis
        # Determine flow regime based on volume
        avg_volume_5d = df.iloc[max(0, i-4):i+1]['volume'].mean()
        avg_volume_20d = df.iloc[max(0, i-19):i+1]['volume'].mean() if i >= 19 else avg_volume_5d
        
        if avg_volume_5d > 1.2 * avg_volume_20d:
            # High flow regime
            factor_value = (flow_imbalance * rejection_momentum + dislocation_factor)
        elif avg_volume_5d < 0.8 * avg_volume_20d:
            # Low flow regime
            factor_value = (flow_imbalance * 0.7 + rejection_momentum * 1.3 + dislocation_factor)
        else:
            # Normal flow
            factor_value = (flow_imbalance + rejection_momentum + dislocation_factor)
        
        result.iloc[i] = factor_value
    
    return result
