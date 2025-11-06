import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Price-Momentum Acceleration with Volume Divergence factor
    
    Combines momentum acceleration (short-term vs medium-term momentum) 
    with volume divergence (volume trend vs price trend) to identify
    potential reversal or continuation signals.
    """
    
    # Calculate Momentum Acceleration
    # Short-Term Momentum (5-day rate of change)
    short_term_momentum = df['close'].pct_change(periods=5)
    
    # Medium-Term Momentum (20-day rate of change)
    medium_term_momentum = df['close'].pct_change(periods=20)
    
    # Momentum Acceleration = Short-term minus Medium-term momentum
    momentum_acceleration = short_term_momentum - medium_term_momentum
    
    # Identify Volume Divergence
    def calculate_trend(series, window):
        """Calculate linear regression slope over a rolling window"""
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y_values = series.iloc[i-window+1:i+1].values
                x_values = np.arange(len(y_values))
                if len(y_values) == window and not np.isnan(y_values).any():
                    slope, _, _, _, _ = linregress(x_values, y_values)
                    slopes.iloc[i] = slope
        return slopes
    
    # Volume Trend (10-day slope)
    volume_trend = calculate_trend(df['volume'], 10)
    
    # Price Trend (10-day slope of close prices)
    price_trend = calculate_trend(df['close'], 10)
    
    # Volume Divergence = Volume Trend / Price Trend
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    volume_divergence = volume_trend / (price_trend + epsilon)
    
    # Combine Signals
    # Multiply Momentum Acceleration by Volume Divergence
    combined_signal = momentum_acceleration * volume_divergence
    
    # Apply sign function to preserve directional information
    final_factor = np.sign(combined_signal) * np.sqrt(np.abs(combined_signal))
    
    return final_factor
