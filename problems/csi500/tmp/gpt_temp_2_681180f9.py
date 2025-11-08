import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily range
    daily_range = df['high'] - df['low']
    
    # Calculate range momentum (5-day difference)
    range_momentum = daily_range - daily_range.shift(5)
    
    # Calculate volume momentum (5-day difference)
    volume_momentum = df['volume'] - df['volume'].shift(5)
    
    # Compute price-volume divergence
    divergence = range_momentum - volume_momentum
    
    # Apply acceleration filter
    divergence_change = divergence - divergence.shift(3)
    divergence_acceleration = divergence_change * np.sign(divergence)
    
    # Calculate gap series
    daily_gap = df['open'] - df['close'].shift(1)
    
    # Compute gap persistence (count of same-sign gaps in past 3 days)
    gap_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i < 3:
            gap_persistence.iloc[i] = 0
        else:
            current_gap_sign = np.sign(daily_gap.iloc[i])
            if current_gap_sign == 0:
                gap_persistence.iloc[i] = 0
            else:
                # Count how many of the previous 3 gaps have the same sign as current gap
                prev_gaps = daily_gap.iloc[i-3:i]
                same_sign_count = (np.sign(prev_gaps) == current_gap_sign).sum()
                gap_persistence.iloc[i] = same_sign_count
    
    # Combine divergence acceleration with gap persistence
    factor = divergence_acceleration * gap_persistence
    
    return factor
