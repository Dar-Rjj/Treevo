import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Volume-Price Divergence Momentum Factor
    Combines volume trend, price momentum, and their divergence patterns
    to generate a predictive alpha factor.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volume Trend Component
    # Calculate 10-day volume trend using linear regression slope
    def volume_trend(volume_series):
        if len(volume_series) < 10:
            return np.nan
        x = np.arange(10)
        slope, _, _, _, _ = stats.linregress(x, volume_series)
        return slope
    
    # Calculate 10-day volume variance
    def volume_variance(volume_series):
        if len(volume_series) < 10:
            return np.nan
        return np.var(volume_series)
    
    # Calculate rolling volume trend and variance
    data['volume_trend'] = data['volume'].rolling(window=10).apply(volume_trend, raw=True)
    data['volume_var'] = data['volume'].rolling(window=10).apply(volume_variance, raw=True)
    
    # Historical average of volume variance (using expanding window)
    data['volume_var_avg'] = data['volume_var'].expanding().mean()
    
    # Price Momentum Component
    # 10-day price momentum
    data['price_momentum'] = (data['close'] / data['close'].shift(10) - 1)
    
    # 10-day price volatility (average daily high-low range)
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['price_volatility'] = data['daily_range'].rolling(window=10).mean()
    
    # Divergence Detection Engine
    conditions = [
        # Strong Volume Weak Price
        (data['volume_trend'] > 0) & (data['price_momentum'] < 0) & (data['volume_var'] > data['volume_var_avg']),
        # Weak Volume Strong Price  
        (data['volume_trend'] < 0) & (data['price_momentum'] > 0) & (data['volume_var'] < data['volume_var_avg']),
        # Synchronized Movement
        ((data['volume_trend'] > 0) & (data['price_momentum'] > 0)) | ((data['volume_trend'] < 0) & (data['price_momentum'] < 0))
    ]
    
    choices = [
        # Strong Volume Weak Price: Negative momentum with volume confirmation
        -1 * data['price_momentum'] * (1 + data['volume_var'] / data['volume_var_avg']),
        # Weak Volume Strong Price: Positive momentum with volume warning
        data['price_momentum'] * (1 - data['volume_var'] / data['volume_var_avg']),
        # Synchronized: Standard momentum score
        data['price_momentum']
    ]
    
    # Generate divergence momentum score
    data['divergence_momentum'] = np.select(conditions, choices, default=data['price_momentum'])
    
    # Volatility-Adjusted Output
    # Scale by price volatility for risk adjustment and combine with volume variance
    volatility_adjustment = 1 / (1 + data['price_volatility'])
    volume_strength = 1 + (data['volume_var'] / data['volume_var_avg'])
    
    # Final divergence momentum factor
    data['factor'] = data['divergence_momentum'] * volatility_adjustment * volume_strength
    
    return data['factor']
