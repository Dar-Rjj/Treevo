import pandas as pd
import pandas as pd

def heuristics_v2(df, n_periods=5):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    close_to_open_return = (df['close'] / df['open']) - 1
    
    # Normalize by Intraday Range and multiply by Close-to-Open Return
    normalized_return = (close_to_open_return / intraday_range) * close_to_open_return
    
    # Integrate Volume: Calculate Volume-Weighted Return
    volume_weighted_return = (close_to_open_return * df['volume']) / df['volume'].rolling(window=n_periods).mean()
    
    # Incorporate Momentum: Calculate Multi-Period Average Returns
    multi_period_returns = df['close'].pct_change().rolling(window=n_periods).mean()
    
    # Calculate Weighted Momentum
    weighted_momentum = multi_period_returns * df['market_cap']  # Assuming 'market_cap' is a column in the DataFrame
    
    # Combine all components to form the final factor
    alpha_factor = (normalized_return + volume_weighted_return + weighted_momentum) / 3
    
    return alpha_factor
