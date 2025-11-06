import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Momentum Reversal with Volume Confirmation factor
    
    Combines short-term price reversal with volume trend confirmation:
    - 4-day reversal from t-5 to t-1
    - 5-day volume slope as confirmation
    - Final factor: Reversal × sign(Volume Slope) × |Volume Slope|
    """
    
    # Calculate 4-day reversal: (Close[t-1] - Close[t-5]) / Close[t-5]
    reversal = (df['close'].shift(1) - df['close'].shift(5)) / df['close'].shift(5)
    
    # Calculate 5-day volume slope using linear regression
    volume_slopes = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i >= 5:  # Need at least 5 days of data
            volumes = df['volume'].iloc[i-5:i].values
            if len(volumes) == 5 and not np.any(np.isnan(volumes)):
                x = np.arange(5)
                slope, _, _, _, _ = stats.linregress(x, volumes)
                volume_slopes.iloc[i] = slope
            else:
                volume_slopes.iloc[i] = np.nan
        else:
            volume_slopes.iloc[i] = np.nan
    
    # Combine signals: Reversal × sign(Volume Slope) × |Volume Slope|
    factor = reversal * np.sign(volume_slopes) * np.abs(volume_slopes)
    
    return factor
