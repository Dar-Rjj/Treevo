import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Price Range
    intraday_price_range = df['high'] - df['low']
    
    # Compute Weighted Intraday Momentum
    intraday_momentum = df['high'] - df['open']
    weighted_intraday_momentum = intraday_momentum * intraday_price_range
    
    # Aggregate Close-to-Previous Close Return
    close_to_prev_close_return = df['close'] - df['close'].shift(1)
    
    # Calculate Volume-Adjusted Return
    volume_adjusted_return = (df['close'] - df['close'].shift(1)) / df['volume']
    volume_adjusted_return = volume_adjusted_return.replace([pd.NP.inf, -pd.NP.inf], 0)  # Handle division by zero
    
    # Incorporate Multi-Day Momentum
    three_day_momentum = df['close'] - df['close'].shift(3)
    five_day_momentum = df['close'] - df['close'].shift(5)
    
    # Combine Factors
    factor = (
        intraday_price_range +
        weighted_intraday_momentum +
        close_to_prev_close_return +
        volume_adjusted_return +
        three_day_momentum +
        five_day_momentum
    )
    
    return factor
