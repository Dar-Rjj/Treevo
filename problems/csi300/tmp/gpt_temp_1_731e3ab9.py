import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate gap efficiency over 5-day window (t-4 to t)
    gap_efficiency_numerator = pd.Series(np.zeros(len(df)), index=df.index)
    gap_efficiency_denominator = pd.Series(np.zeros(len(df)), index=df.index)
    
    for i in range(4, len(df)):
        window_sum_numerator = 0
        window_sum_denominator = 0
        
        for j in range(i-4, i+1):
            # Gap: |Open_j - Close_{j-1}|
            if j > 0:  # Ensure we have previous close
                gap = abs(df['open'].iloc[j] - df['close'].iloc[j-1])
                window_sum_numerator += gap
            
            # Price range denominator: max(High_j - Low_j, |High_j - Open_j|, |Low_j - Open_j|)
            high_low_range = df['high'].iloc[j] - df['low'].iloc[j]
            high_open_range = abs(df['high'].iloc[j] - df['open'].iloc[j])
            low_open_range = abs(df['low'].iloc[j] - df['open'].iloc[j])
            denominator_component = max(high_low_range, high_open_range, low_open_range)
            window_sum_denominator += denominator_component
        
        gap_efficiency_numerator.iloc[i] = window_sum_numerator
        gap_efficiency_denominator.iloc[i] = window_sum_denominator
    
    # Calculate gap efficiency ratio
    gap_efficiency = gap_efficiency_numerator / gap_efficiency_denominator
    gap_efficiency = gap_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Calculate volume-adjusted return
    volume_adjusted_return = pd.Series(np.zeros(len(df)), index=df.index)
    for i in range(1, len(df)):
        # (Close_t / Open_t - 1) × log(Volume_t / Volume_{t-1})
        price_return = df['close'].iloc[i] / df['open'].iloc[i] - 1
        volume_ratio = df['volume'].iloc[i] / df['volume'].iloc[i-1]
        volume_adjusted_return.iloc[i] = price_return * np.log(volume_ratio)
    
    # Combine gap efficiency with volume-adjusted return
    factor = gap_efficiency * volume_adjusted_return
    
    return factor
