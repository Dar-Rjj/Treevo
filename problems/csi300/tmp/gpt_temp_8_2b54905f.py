import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing

def heuristics_v2(df):
    # Calculate Intraday Volatility
    df['intraday_volatility'] = df['high'] - df['low']
    
    # Weight by Volume
    df['weighted_volatility'] = df['intraday_volatility'] * df['volume']
    
    # Calculate Intraday Momentum
    df['close_open_diff'] = df['close'] - df['open']
    df['momentum'] = df['close_open_diff'].rolling(window=5).sum()
    df['positive_momentum'] = df['momentum'].apply(lambda x: max(x, 0))
    
    # Integrate Momentum and Weighted Volatility
    df['integrated_factor'] = df['weighted_volatility'] + df['positive_momentum']
    
    # Apply Exponential Smoothing
    alpha = 0.2
    df['smoothed_factor'] = df['integrated_factor'].ewm(alpha=alpha, adjust=False).mean()
    
    # Ensure values are positive
    df['smoothed_factor_positive'] = df['smoothed_factor'].apply(lambda x: max(x, 0))
    
    # Apply Logarithmic Transformation
    df['log_smoothed_factor'] = np.log1p(df['smoothed_factor_positive'])
    
    return df['log_smoothed_factor']
