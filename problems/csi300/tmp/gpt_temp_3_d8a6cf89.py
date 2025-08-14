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
    df['momentum'] = df['close'] - df['open']
    df['momentum_rolling_sum'] = df['momentum'].rolling(window=5).sum()
    
    # Integrate Momentum and Weighted Volatility
    df['integrated_factor'] = df['weighted_volatility'] + df['momentum_rolling_sum']
    
    # Apply Exponential Smoothing
    alpha = 0.6
    smoothed_factor = ExponentialSmoothing(df['integrated_factor'], trend='add', initialization_method='heuristic').fit(smoothing_level=alpha, optimized=False)
    df['smoothed_factor'] = smoothed_factor.fittedvalues
    
    # Ensure values are positive before logarithmic transformation
    df['log_transformed_factor'] = np.log1p(df['smoothed_factor'].abs())
    
    # Factor Output
    factor_values = df['log_transformed_factor']
    
    return factor_values
