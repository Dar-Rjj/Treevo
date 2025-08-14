import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Simple Momentum
    n = 10  # Lookback period for momentum
    simple_momentum = df['close'] / df['close'].shift(n) - 1
    
    # Volume Adjusted Component
    daily_volume_change = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    volume_adjusted_momentum = simple_momentum * daily_volume_change
    
    # Enhanced Price Reversal Sensitivity
    high_low_spread = (df['high'] - df['low']) / df['volume']
    open_close_spread = (df['open'] - df['close']) / df['volume']
    weighted_high_low_spread = high_low_spread * df['volume']
    weighted_open_close_spread = open_close_spread * df['volume']
    combined_spread = (weighted_high_low_spread + weighted_open_close_spread) / 2
    
    # Volume Trend Component
    m = 20  # Lookback period for moving average of volume
    volume_ma = df['volume'].rolling(window=m).mean()
    volume_trend = df['volume'] / volume_ma - 1
    
    # Volatility Adjustment
    true_range = (df['high'] - df['low']).abs().max(axis=1)
    k = 20  # Lookback period for moving average of true range
    true_range_ma = true_range.rolling(window=k).mean()
    
    # Combine Components
    factor = (
        volume_adjusted_momentum
        - combined_spread
        + volume_trend
    ) / true_range_ma
    
    return factor
