import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = df['close'] / df['open'] - 1
    
    # Calculate Enhanced Intraday High-Low Spread
    high_low_spread = (df['high'] - df['low']) * df['volume']
    
    # Combine Intraday High-Low Spread and Intraday Return
    combined_intraday_factor = high_low_spread * intraday_return
    
    # Incorporate Volume and Amount Influence
    n_day = 20
    avg_volume = df['volume'].rolling(window=n_day).mean()
    volume_impact = df['volume'] / avg_volume
    amount_impact = df['amount'] / avg_volume
    combined_volume_amount_impact = volume_impact + amount_impact
    weighted_combined_intraday_factor = combined_intraday_factor * combined_volume_amount_impact
    
    # Calculate True Range for each day
    prev_close = df['close'].shift(1)
    true_range = np.maximum.reduce([df['high'] - df['low'], abs(df['high'] - prev_close), abs(df['low'] - prev_close)])
    
    # Calculate 14-day Simple Moving Average of the True Range
    sma_true_range = true_range.rolling(window=14).mean()
    
    # Construct the Momentum Component
    momentum_component = (df['close'] - sma_true_range) / sma_true_range
    
    # Enhance with Volume-Weighted High-Low Difference
    high_low_diff = df['high'] - df['low']
    volume_weighted_high_low_diff = high_low_diff * df['volume']
    
    # Calculate Daily Log Returns
    daily_log_returns = np.log(df['close'] / df['close'].shift(1))
    
    # Compute Realized Volatility
    realized_volatility = daily_log_returns.rolling(window=20).std()
    
    # Normalize Momentum by Volatility
    normalized_momentum = momentum_component / realized_volatility
    
    # Introduce Trend Component
    trend_50ma = df['close'].rolling(window=50).mean()
    trend_direction = np.where(df['close'] > trend_50ma, 1, -1)
    
    # Calculate 14-Day Volume-Weighted Intraday Return
    intraday_returns = df['close'] - df['open']
    volume_weighted_intraday_return = (intraday_returns * df['volume']).rolling(window=14).sum() / df['volume'].rolling(window=14).sum()
    
    # Adjust for Volatility and Price Trend
    combined_value = normalized_momentum + weighted_combined_intraday_factor
    adjusted_value = combined_value * trend_direction
    
    # Synthesize Alpha Factor
    alpha_factor = (momentum_component + weighted_combined_intraday_factor + 
                    volume_weighted_intraday_return + volume_weighted_high_low_diff)
    
    return alpha_factor
