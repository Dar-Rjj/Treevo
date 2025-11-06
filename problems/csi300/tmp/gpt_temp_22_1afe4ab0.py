import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required shifts for momentum calculations
    for i in range(len(df)):
        if i < 6:  # Need at least 6 days of history
            result.iloc[i] = 0
            continue
            
        current = df.iloc[i]
        # Get historical data points
        close_t2 = df.iloc[i-2]['close']
        close_t3 = df.iloc[i-3]['close']
        close_t5 = df.iloc[i-5]['close']
        close_t6 = df.iloc[i-6]['close']
        
        volume_t2 = df.iloc[i-2]['volume']
        volume_t5 = df.iloc[i-5]['volume']
        
        amount_t2 = df.iloc[i-2]['amount']
        amount_t5 = df.iloc[i-5]['amount']
        
        high_t1 = df.iloc[i-1]['high']
        low_t1 = df.iloc[i-1]['low']
        
        # Momentum Acceleration Framework
        short_term_accel = (current['close']/close_t2 - 1) - (current['close']/close_t5 - 1)
        medium_term_accel = (current['close']/close_t3 - 1) - (current['close']/close_t6 - 1)
        core_price_accel = (short_term_accel + medium_term_accel) / 2
        
        # Volume-Amount Acceleration
        volume_accel = (current['volume']/volume_t2 - 1) - (current['volume']/volume_t5 - 1)
        amount_accel = (current['amount']/amount_t2 - 1) - (current['amount']/amount_t5 - 1)
        
        # Range Efficiency Analysis
        price_efficiency = (current['close'] - current['open']) / (current['high'] - current['low'])
        if (high_t1 - low_t1) == 0:
            range_expansion_ratio = 1.0
        else:
            range_expansion_ratio = (current['high'] - current['low']) / (high_t1 - low_t1)
        
        # Multi-dimensional Integration
        acceleration_convergence = core_price_accel * volume_accel * amount_accel
        efficiency_score = price_efficiency * range_expansion_ratio
        intraday_performance = (current['close'] - current['open']) / current['open']
        
        # Composite Alpha Generation
        core_factor = acceleration_convergence * efficiency_score
        final_signal = core_factor * intraday_performance * abs(core_factor * intraday_performance)
        
        result.iloc[i] = final_signal
    
    return result
