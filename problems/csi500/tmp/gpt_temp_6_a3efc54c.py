import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum Divergence with Volume Confirmation alpha factor
    
    This factor captures acceleration/deceleration patterns in price momentum
    confirmed by volume trends and volume-price correlation.
    """
    # Price Momentum Component
    close = df['close']
    
    # Calculate Short-Term Momentum (5-day price change)
    short_term_momentum = close.pct_change(periods=5)
    
    # Calculate Medium-Term Momentum (20-day price change)
    medium_term_momentum = close.pct_change(periods=20)
    
    # Calculate Momentum Divergence (acceleration/deceleration)
    momentum_divergence = short_term_momentum - medium_term_momentum
    
    # Volume Confirmation Component
    volume = df['volume']
    
    # Calculate Volume Trend (20-day volume slope)
    def volume_slope(vol_series):
        if len(vol_series) < 2:
            return np.nan
        x = np.arange(len(vol_series))
        return np.polyfit(x, vol_series, 1)[0] / np.mean(vol_series)
    
    volume_trend = volume.rolling(window=20).apply(volume_slope, raw=False)
    
    # Calculate Volume-Price Correlation (20-day rolling correlation)
    volume_price_corr = close.rolling(window=20).corr(volume)
    
    # Combine components to generate final alpha factor
    # Multiply momentum divergence by volume trend and adjust by correlation
    alpha_factor = momentum_divergence * volume_trend * volume_price_corr
    
    return alpha_factor
