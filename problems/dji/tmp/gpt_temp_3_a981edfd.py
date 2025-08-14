import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Spread
    df['high_low_spread'] = df['high'] - df['low']
    
    # Divide by Previous Close
    df['momentum'] = df['high_low_spread'] / df['close'].shift(1)
    
    # Calculate Price Momentum
    df['price_momentum'] = df['close'] - df['close'].shift(10)
    
    # Calculate Intraday Returns
    df['intraday_return'] = (df['high'] - df['low']) / df['low']
    
    # Adjust by Volume
    N = 5
    df['volume_sum'] = df['volume'].rolling(window=N).sum()
    df['avg_volume'] = df['volume_sum'] / N
    df['volume_adjusted_momentum'] = df['momentum'] * df['avg_volume']
    
    # Calculate Cumulative Volume-Weighted Momentum
    df['cum_vol_weighted_mom'] = (df['volume_adjusted_momentum'] * df['volume']).rolling(window=10).sum()
    
    # Aggregate Momentum Over M Days
    M = 10
    df['agg_momentum'] = df['cum_vol_weighted_mom'].rolling(window=M).mean()
    
    # Confirm with Volume Trend
    avg_volume_past_10_days = df['volume'].rolling(window=10).mean()
    current_volume = df['volume']
    volume_ratio = current_volume / avg_volume_past_10_days
    
    conditions = [
        (volume_ratio > 1.2),
        (volume_ratio <= 1.2)
    ]
    choices = [
        (df['price_momentum'] * df['intraday_return']) * df['agg_momentum'],
        0.5 * (df['price_momentum'] + df['intraday_return'])
    ]
    
    df['alpha_factor'] = pd.np.select(conditions, choices, default=0)
    
    return df['alpha_factor']
