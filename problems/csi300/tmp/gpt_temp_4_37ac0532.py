import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Calculate Intraday Volatility
    intraday_volatility = df['high'] - df['low']
    
    # Weight by Volume
    weighted_volatility = intraday_volatility * df['volume']
    
    # Calculate Intraday Momentum
    daily_momentum = df['close'] - df['open']
    rolling_momentum = daily_momentum.rolling(window=10).sum()
    
    # Integrate Momentum and Volatility
    integrated_factor = weighted_volatility + rolling_momentum
    
    # Define Initial Smoothing Factor
    initial_alpha = 0.9
    smoothed_factor = integrated_factor.ewm(alpha=initial_alpha, adjust=False).mean()
    
    # Adjust Smoothing Factor Based on Recent Data Stability
    recent_stability = integrated_factor.rolling(window=10).std()
    alpha_adjustment = 0.1 * (1 - zscore(recent_stability))
    adjusted_smoothed_factor = smoothed_factor.ewm(alpha=alpha_adjustment, adjust=False).mean()
    
    # Ensure Values are Positive
    positive_factor = adjusted_smoothed_factor + 1e-6
    
    # Apply Logarithmic Transformation
    log_transformed_factor = np.log(positive_factor)
    
    return log_transformed_factor
