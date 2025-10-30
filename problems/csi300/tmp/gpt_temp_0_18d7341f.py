import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(4, len(df)):
        # Calculate Gap Efficiency components
        gap_sum = 0.0
        volatility_sum = 0.0
        
        for j in range(i-4, i+1):
            # Gap: |Open_j - Close_{j-1}|
            if j > 0:
                gap = abs(df['open'].iloc[j] - df['close'].iloc[j-1])
            else:
                gap = 0.0
            gap_sum += gap
            
            # Volatility: max(High_j - Low_j, |High_j - Open_j|, |Low_j - Open_j|)
            vol1 = df['high'].iloc[j] - df['low'].iloc[j]
            vol2 = abs(df['high'].iloc[j] - df['open'].iloc[j])
            vol3 = abs(df['low'].iloc[j] - df['open'].iloc[j])
            volatility_sum += max(vol1, vol2, vol3)
        
        # Gap Ratio
        if volatility_sum > 0:
            gap_ratio = gap_sum / volatility_sum
        else:
            gap_ratio = 0.0
        
        # Volatility Adjustment
        close_window = df['close'].iloc[i-4:i+1]
        if len(close_window) > 1 and close_window.mean() > 0:
            vol_adjustment = gap_ratio / (close_window.std() / close_window.mean())
        else:
            vol_adjustment = 0.0
        
        # Volume Acceleration components
        # Volume-Weighted Gap Return
        if df['open'].iloc[i] > 0 and df['volume'].iloc[i-1] > 0:
            gap_return = (df['close'].iloc[i] / df['open'].iloc[i] - 1)
            volume_ratio = np.log(df['volume'].iloc[i] / df['volume'].iloc[i-1])
            volume_weighted_gap_return = gap_return * volume_ratio
        else:
            volume_weighted_gap_return = 0.0
        
        # Volume-Gap Alignment
        if i > 0:
            gap_direction = np.sign(df['open'].iloc[i] - df['close'].iloc[i-1])
            volume_direction = np.sign(df['volume'].iloc[i] - df['volume'].iloc[i-1])
            volume_gap_alignment = gap_direction * volume_direction
        else:
            volume_gap_alignment = 0.0
        
        # Final Alpha Factor
        result.iloc[i] = vol_adjustment * volume_weighted_gap_return * volume_gap_alignment
    
    return result
