import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, N=20, M=20):
    # Obtain Close Prices
    close_prices = df['close']
    
    # Compute Log Return over N Days
    log_returns = np.log(close_prices) - np.log(close_prices.shift(1))
    
    # Calculate Cumulative Sum of Log Returns
    momentum = log_returns.rolling(window=N).sum()
    
    # Define Upper and Lower Thresholds
    upper_threshold = momentum.quantile(0.9)
    lower_threshold = momentum.quantile(0.1)
    
    # Clip Log Returns to Thresholds
    momentum_clipped = np.clip(momentum, lower_threshold, upper_threshold)
    
    # Retrieve Volumes
    volumes = df['volume']
    
    # Calculate Volume Relative to Mean
    volume_relative = volumes / volumes.rolling(window=N).mean()
    
    # Incorporate Volume into Momentum
    volume_adjusted_momentum = momentum_clipped * volume_relative
    
    # Determine Absolute Price Changes
    absolute_price_changes = abs(close_prices - close_prices.shift(1))
    
    # Compute Standard Deviation of Absolute Price Changes Over M Days
    std_dev_volatility = absolute_price_changes.rolling(window=M).std()
    
    # Calculate Exponential Moving Average (EMA) of Absolute Price Changes
    ema_volatility = absolute_price_changes.ewm(span=M).mean()
    
    # Calculate Inter-Quartile Range (IQR) of Absolute Price Changes
    iqr_volatility = absolute_price_changes.rolling(window=M).quantile(0.75) - absolute_price_changes.rolling(window=M).quantile(0.25)
    
    # Combine Weighted Momentum, Volume, and Volatility Components
    weights = [0.4, 0.3, 0.3]  # Weights for momentum, volume, and volatility
    combined_factor = (weights[0] * volume_adjusted_momentum + 
                       weights[1] * (std_dev_volatility + ema_volatility + iqr_volatility) / 3 +
                       weights[2] * (volume_relative - 1))
    
    # Integrate Market Breadth and Key Economic Indicators
    # For simplicity, we assume the following indicators are available in the DataFrame
    advance_decline_line = df['advance_decline_line']
    new_highs = df['new_highs']
    new_lows = df['new_lows']
    interest_rates = df['interest_rates']
    gdp_growth_rate = df['gdp_growth_rate']
    
    # Calculate Advance-Decline Line
    breadth_indicator = (advance_decline_line + new_highs - new_lows) / 3
    
    # Incorporate Key Economic Indicators
    economic_indicator = (interest_rates + gdp_growth_rate) / 2
    
    # Final Factor Calculation
    final_factor = combined_factor * (breadth_indicator + economic_indicator) / 2
    
    return final_factor
