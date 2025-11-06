import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate 5-day price momentum using Close
    momentum_5d = df['close'] / df['close'].shift(5) - 1
    
    # Compute momentum change from previous period (acceleration)
    momentum_acceleration = momentum_5d - momentum_5d.shift(1)
    
    # Calculate 5-day volume trend slope using linear regression
    volume_slope = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        if i >= 5:
            window_vol = df['volume'].iloc[i-4:i+1]
            if not window_vol.isna().any():
                x = np.arange(5)
                y = window_vol.values
                slope = np.polyfit(x, y, 1)[0]
                volume_slope.iloc[i] = slope / window_vol.mean()  # Normalize by mean volume
    
    # Combine acceleration with volume slope
    factor = momentum_acceleration * volume_slope
    
    return factor
