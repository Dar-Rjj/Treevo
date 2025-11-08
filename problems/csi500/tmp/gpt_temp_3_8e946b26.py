import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum Analysis
    # Short-term momentum (5-day return)
    data['short_momentum'] = data['close'].pct_change(periods=5)
    
    # Medium-term momentum (20-day return)
    data['medium_momentum'] = data['close'].pct_change(periods=20)
    
    # Volume Trend Analysis
    # 5-day volume slope using linear regression
    def volume_slope(volume_series):
        if len(volume_series) < 5:
            return np.nan
        x = np.arange(5)
        y = volume_series.values
        return np.polyfit(x, y, 1)[0] / np.mean(y)  # Normalize by mean volume
    
    data['volume_slope'] = data['volume'].rolling(window=5, min_periods=5).apply(
        volume_slope, raw=False
    )
    
    # Signal Generation
    # Momentum divergence detection
    data['momentum_divergence'] = data['short_momentum'] - data['medium_momentum']
    
    # Volume confirmation check
    # Positive momentum divergence with increasing volume is bullish
    # Negative momentum divergence with decreasing volume is bearish
    data['factor'] = data['momentum_divergence'] * data['volume_slope']
    
    # Return the factor series
    return data['factor']
