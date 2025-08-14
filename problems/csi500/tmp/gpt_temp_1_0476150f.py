import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Price Difference
    intraday_price_diff = df['high'] - df['low']
    
    # Calculate Close to Open Ratio
    close_to_open_ratio = df['close'] / df['open']
    
    # Calculate Volume-Weighted Intraday Return
    volume_weighted_intraday_return = df['volume'] * (close_to_open_ratio - 1)
    
    # Calculate 5-Day Simple Moving Average (SMA) of Close
    sma_5_day = df['close'].rolling(window=5).mean()
    
    # Calculate 5-Day Momentum
    momentum_5_day = df['close'] - sma_5_day
    
    # Integrate Intraday and 5-Day Momentum
    integrated_momentum = volume_weighted_intraday_return + momentum_5_day
    
    # Apply Market Cap Adjustment (assuming 'market_cap' is a column in the DataFrame)
    final_alpha_factor = integrated_momentum / df['market_cap']
    
    return final_alpha_factor
