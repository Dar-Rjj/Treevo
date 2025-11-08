import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Momentum Divergence with Volume Trend Confirmation factor
    
    Combines short-term vs medium-term momentum divergence with 
    volume trend confirmation to identify strong directional signals.
    """
    # Calculate Short-Term Momentum (3-day return)
    short_term_momentum = df['close'] / df['close'].shift(3) - 1
    
    # Calculate Medium-Term Momentum (10-day return)
    medium_term_momentum = df['close'] / df['close'].shift(10) - 1
    
    # Calculate Momentum Divergence
    momentum_divergence = short_term_momentum - medium_term_momentum
    
    # Calculate Volume Trend (5-day slope)
    volume_trend = pd.Series(index=df.index, dtype=float)
    
    for i in range(4, len(df)):
        if i >= 4:
            volume_window = df['volume'].iloc[i-4:i+1]
            if len(volume_window) == 5 and not volume_window.isna().any():
                x = np.arange(5)
                slope, _, _, _, _ = linregress(x, volume_window.values)
                volume_trend.iloc[i] = slope
            else:
                volume_trend.iloc[i] = np.nan
    
    # Combine signals
    factor = momentum_divergence * volume_trend
    
    return factor
