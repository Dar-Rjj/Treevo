import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Divergence Momentum Factor
    Combines volatility-normalized momentum across timeframes with volume-price divergence detection
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Volatility-Normalized Price Signals
    close = df['close']
    volume = df['volume']
    
    # Calculate returns
    returns_5d = close.pct_change(5)
    returns_20d = close.pct_change(20)
    returns_60d = close.pct_change(60)
    
    # Calculate volatilities
    vol_20d = close.pct_change().rolling(20).std()
    vol_60d = close.pct_change().rolling(60).std()
    vol_120d = close.pct_change().rolling(120).std()
    
    # Volatility-normalized momentum
    mom_short = returns_5d / vol_20d.replace(0, np.nan)
    mom_medium = returns_20d / vol_60d.replace(0, np.nan)
    mom_long = returns_60d / vol_120d.replace(0, np.nan)
    
    # Volume-Price Divergence Detection
    def calculate_volume_slope(series, window):
        """Calculate linear regression slope of volume"""
        x = np.arange(window)
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
                continue
            y = series.iloc[i-window+1:i+1].values
            if len(y) == window:
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(slope)
            else:
                slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    # Volume slopes
    vol_slope_5d = calculate_volume_slope(volume, 5)
    vol_slope_20d = calculate_volume_slope(volume, 20)
    vol_slope_60d = calculate_volume_slope(volume, 60)
    
    # Volume-price divergence (positive when volume trend confirms price momentum)
    divergence_short = np.sign(mom_short) * np.sign(vol_slope_5d)
    divergence_medium = np.sign(mom_medium) * np.sign(vol_slope_20d)
    divergence_long = np.sign(mom_long) * np.sign(vol_slope_60d)
    
    # Volume persistence filter (favor consistent volume trends)
    vol_persistence = (
        (vol_slope_5d.rolling(3).std() < vol_slope_5d.rolling(10).std()) &
        (vol_slope_20d.rolling(5).std() < vol_slope_20d.rolling(20).std())
    ).astype(float)
    
    # Combined Divergence Factor
    # Weight divergences by timeframe (more weight to medium-term)
    weights = pd.DataFrame({
        'short': 0.3,
        'medium': 0.5,
        'long': 0.2
    }, index=divergence_short.index)
    
    combined_divergence = (
        weights['short'] * divergence_short +
        weights['medium'] * divergence_medium +
        weights['long'] * divergence_long
    )
    
    # Apply volume persistence filter
    final_factor = combined_divergence * vol_persistence
    
    # Normalize the factor
    final_factor = (final_factor - final_factor.rolling(60).mean()) / final_factor.rolling(60).std()
    
    return final_factor
