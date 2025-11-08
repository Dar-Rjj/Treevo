import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Dynamic Volume-Weighted Momentum Divergence alpha factor
    Combines price momentum with volume trends to detect divergences
    """
    close = df['close']
    volume = df['volume']
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate momentum slopes using linear regression
    def calc_momentum_slope(price_series, window):
        slopes = pd.Series(index=price_series.index, dtype=float)
        for i in range(window-1, len(price_series)):
            if i >= window-1:
                y = price_series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes.iloc[i] = slope
        return slopes
    
    # Calculate volume trend slopes
    def calc_volume_slope(vol_series, window):
        slopes = pd.Series(index=vol_series.index, dtype=float)
        for i in range(window-1, len(vol_series)):
            if i >= window-1:
                y = vol_series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes.iloc[i] = slope
        return slopes
    
    # Short-term momentum (3-day)
    mom_3d_slope = calc_momentum_slope(close, 3)
    mom_3d_return = close.pct_change(3)
    
    # Medium-term momentum (8-day)
    mom_8d_slope = calc_momentum_slope(close, 8)
    mom_8d_return = close.pct_change(8)
    
    # Volume trends
    vol_3d_slope = calc_volume_slope(volume, 3)
    vol_3d_mean = volume.rolling(window=3).mean()
    
    vol_8d_slope = calc_volume_slope(volume, 8)
    vol_8d_mean = volume.rolling(window=8).mean()
    
    # Momentum divergence detection
    momentum_divergence = mom_3d_slope - mom_8d_slope
    momentum_acceleration = momentum_divergence.diff()
    
    # Volume confirmation analysis
    volume_trend_divergence = vol_3d_slope - vol_8d_slope
    volume_momentum_alignment = np.sign(mom_3d_slope) * np.sign(vol_3d_slope)
    
    # Dynamic signal weighting
    current_volume_weight = volume / volume.rolling(window=20).mean()
    
    # Recent volume regime (5-day volatility)
    volume_volatility = volume.rolling(window=5).std() / volume.rolling(window=5).mean()
    volume_stability_weight = 1.0 / (1.0 + volume_volatility)
    
    # Combine signals with dynamic weighting
    weighted_divergence = (
        momentum_divergence * 
        volume_momentum_alignment * 
        current_volume_weight * 
        volume_stability_weight
    )
    
    # Apply momentum-direction bias
    momentum_bias = np.sign(mom_3d_slope)
    final_signal = weighted_divergence * momentum_bias
    
    # Normalize the final signal
    alpha = (final_signal - final_signal.rolling(window=20).mean()) / final_signal.rolling(window=20).std()
    
    return alpha.fillna(0)
