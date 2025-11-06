import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Momentum-Adjusted Volume Divergence factor that combines price momentum 
    with volume momentum to identify volume-confirmed price trend strength.
    """
    close = df['close']
    volume = df['volume']
    
    # Calculate Price Momentum
    # Short-term return (5-day)
    short_term_return = close / close.shift(5) - 1
    
    # Medium-term return (20-day)
    medium_term_return = close / close.shift(20) - 1
    
    # Combined price momentum (weighted average)
    price_momentum = 0.6 * short_term_return + 0.4 * medium_term_return
    
    # Calculate Volume Momentum
    # Volume ratio: current volume vs 20-day average
    volume_avg_20 = volume.rolling(window=20, min_periods=10).mean()
    volume_ratio = volume / volume_avg_20
    
    # Volume trend: linear slope of last 10 days
    def calc_volume_slope(volume_series):
        if len(volume_series) < 10:
            return np.nan
        x = np.arange(len(volume_series))
        slope, _, _, _, _ = linregress(x, volume_series)
        return np.sign(slope)
    
    volume_trend = volume.rolling(window=10, min_periods=5).apply(
        calc_volume_slope, raw=False
    )
    
    # Volume momentum: combine ratio and trend
    volume_momentum = volume_ratio * volume_trend
    
    # Combine Signals
    combined_signal = price_momentum * volume_momentum
    
    # Apply Non-Linear Transformation
    # Scale by empirical constant to normalize the distribution
    scaling_factor = 2.0
    final_factor = np.tanh(combined_signal * scaling_factor)
    
    return pd.Series(final_factor, index=df.index, name='momentum_adjusted_volume_divergence')
