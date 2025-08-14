import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing

def heuristics_v2(df, momentum_timeframe=5, smoothing_alpha=0.1):
    # Calculate Intraday Volatility
    intraday_volatility = df['high'] - df['low']
    
    # Weight by Volume
    weighted_volatility = intraday_volatility * df['volume']
    
    # Calculate Intraday Momentum
    daily_momentum = df['close'] - df['open']
    intraday_momentum = daily_momentum.rolling(window=momentum_timeframe).sum()
    
    # Integrate Momentum and Volatility
    integrated_value = weighted_volatility + intraday_momentum
    
    # Apply Exponential Smoothing
    smoothed_integrated_value = integrated_value.ewm(alpha=smoothing_alpha, adjust=False).mean()
    
    # Ensure values are positive before applying logarithmic transformation
    smoothed_integrated_value[smoothed_integrated_value <= 0] = 1e-10  # small positive value to avoid log(0)
    
    # Apply Logarithmic Transformation
    factor_values = np.log(smoothed_integrated_value)
    
    return factor_values
