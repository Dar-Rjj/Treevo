import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df, window=20):
    """
    Volatility-Adjusted Volume-Price Divergence factor
    Combines volatility measures, volume momentum, and price-volume divergence
    """
    # Make a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Volatility Component
    # Daily Range Volatility
    daily_range = data['high'] - data['low']
    range_volatility = daily_range.rolling(window=window).std()
    
    # Returns Volatility
    returns = data['close'].pct_change()
    returns_volatility = returns.rolling(window=window).std()
    
    # Combined Volatility (equal weighting)
    combined_volatility = (range_volatility + returns_volatility) / 2
    
    # Volume Component
    # Volume Momentum
    volume_mean = data['volume'].rolling(window=window).mean()
    volume_momentum = data['volume'] / volume_mean
    
    # Trade Size (average price per share)
    trade_size = data['amount'] / data['volume']
    
    # Divergence Signal
    # Volume Trend Slope
    def calc_slope(series, window):
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                slope, _, _, _, _ = linregress(x, y)
                slopes.append(slope)
        return pd.Series(slopes, index=series.index)
    
    volume_slope = calc_slope(data['volume'], window)
    price_slope = calc_slope(data['close'], window)
    
    # Divergence direction
    divergence_sign = np.sign(volume_slope - price_slope)
    
    # Final Factor Calculation
    # (Volume Momentum / Volatility) × Trade Size × Sign(Volume Slope - Price Slope)
    volatility_adjusted_volume = volume_momentum / combined_volatility
    factor = volatility_adjusted_volume * trade_size * divergence_sign
    
    # Handle edge cases
    factor = factor.replace([np.inf, -np.inf], np.nan)
    
    return factor
