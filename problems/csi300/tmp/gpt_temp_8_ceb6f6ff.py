import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Volatility
    intraday_volatility = df['high'] - df['low']
    
    # Weight by Volume
    weighted_volatility = intraday_volatility * df['volume']
    
    # Calculate Intraday Momentum
    intraday_momentum = df['close'] - df['open']
    
    # Integrate Momentum and Volatility
    sum_weighted_volatility_momentum = weighted_volatility + intraday_momentum
    
    # Apply Exponential Smoothing
    alpha = 0.2
    smoothed_value = sum_weighted_volatility_momentum.copy()
    for i in range(1, len(smoothed_value)):
        smoothed_value.iloc[i] = alpha * smoothed_value.iloc[i] + (1 - alpha) * smoothed_value.iloc[i-1]
    
    # Apply Logarithmic Transformation
    log_smoothed_value = np.log(smoothed_value)
    
    # Integrate Volume Dynamically
    volume_change = df['volume'].diff().fillna(0)
    integrated_value = log_smoothed_value * volume_change
    
    # Apply Exponential Smoothing to Integrated Value
    alpha = 0.1
    final_smoothed_value = integrated_value.copy()
    for i in range(1, len(final_smoothed_value)):
        final_smoothed_value.iloc[i] = alpha * final_smoothed_value.iloc[i] + (1 - alpha) * final_smoothed_value.iloc[i-1]
    
    # Consider Additional Momentum Terms
    short_term_momentum = df['close'].rolling(window=5).mean().fillna(method='bfill')
    long_term_momentum = df['close'].rolling(window=20).mean().fillna(method='bfill')
    
    # Final Integration
    final_integrated_value = (short_term_momentum + long_term_momentum) + final_smoothed_value
    
    # Apply Logarithmic Transformation
    final_factor = np.log(final_integrated_value)
    
    return final_factor
