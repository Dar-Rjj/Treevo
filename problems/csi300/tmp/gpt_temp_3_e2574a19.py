import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Volatility
    df['intraday_volatility'] = df['high'] - df['low']
    
    # Weight by Volume
    df['weighted_volatility'] = df['intraday_volatility'] * df['volume']
    
    # Calculate Intraday Momentum
    df['close_open_diff'] = df['close'] - df['open']
    rolling_window = 5  # You can adjust the window size
    df['momentum'] = df['close_open_diff'].rolling(window=rolling_window).sum()
    
    # Integrate Momentum and Volatility
    df['integrated_factor'] = df['weighted_volatility'] + df['momentum']
    
    # Define Adaptive Smoothing Factor Based on Recent Volatility
    recent_volatility = df['intraday_volatility'].rolling(window=5).mean()
    adaptive_smoothing_factor = 1 / (1 + recent_volatility)
    
    # Apply Exponential Smoothing Formula
    alpha = adaptive_smoothing_factor
    smoothed_factor = df['integrated_factor'].ewm(alpha=alpha, ignore_na=True).mean()
    
    # Ensure Values are Positive
    small_constant = 1e-6
    smoothed_factor_positive = smoothed_factor + small_constant
    
    # Apply Logarithmic Transformation
    log_smoothed_factor = np.log(smoothed_factor_positive)
    
    # Incorporate Sector-Specific Momentum
    # Assuming 'sector' is a column in the DataFrame
    sector_momentum = df.groupby('sector')['momentum'].transform('mean')
    factor_with_sector = log_smoothed_factor + sector_momentum
    
    return factor_with_sector
