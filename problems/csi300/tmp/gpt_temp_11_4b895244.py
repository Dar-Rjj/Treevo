import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Price and Volume Divergence Momentum factor
    Combines price momentum with volume divergence signals, scaled by volatility
    """
    # Make a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Price Momentum Component
    # Recent price trend using High prices (5 days)
    high_5d_slope = data['high'].rolling(window=5).apply(
        lambda x: stats.linregress(range(5), x)[0] if len(x) == 5 else np.nan
    )
    
    # Medium-term price trend using Low prices (20 days)
    low_20d_slope = data['low'].rolling(window=20).apply(
        lambda x: stats.linregress(range(20), x)[0] if len(x) == 20 else np.nan
    )
    
    # Combined price momentum (weight recent trend more heavily)
    price_momentum = 0.6 * high_5d_slope + 0.4 * low_20d_slope
    
    # Volume Divergence Component
    # Volume trend (10 days)
    volume_slope = data['volume'].rolling(window=10).apply(
        lambda x: stats.linregress(range(10), x)[0] if len(x) == 10 else np.nan
    )
    
    # Price trend for comparison (10 days using close prices)
    price_slope_10d = data['close'].rolling(window=10).apply(
        lambda x: stats.linregress(range(10), x)[0] if len(x) == 10 else np.nan
    )
    
    # Volume divergence: ratio of volume slope to price slope with sign preservation
    volume_divergence = np.sign(volume_slope) * np.where(
        price_slope_10d != 0,
        np.abs(volume_slope / price_slope_10d),
        np.sign(volume_slope) * np.abs(volume_slope)
    )
    
    # Combine Components
    combined_signal = price_momentum * volume_divergence
    
    # Scale by Recent Volatility
    # Calculate returns
    returns = data['close'].pct_change()
    
    # Volatility (10-day standard deviation of returns)
    volatility = returns.rolling(window=10).std()
    
    # Final factor: combined signal divided by volatility (avoid division by zero)
    factor = combined_signal / np.where(volatility != 0, volatility, 1)
    
    return factor
