import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close to High-Low Spread
    high_low_spread = df['High'] - df['Low']
    close_to_high_low_spread = (df['Close'] - df['Low']) / high_low_spread
    
    # Calculate Volume-Weighted Momentum
    previous_close = df['Close'].shift(1)
    momentum = df['Close'] - previous_close
    volume_weighted_momentum = momentum * np.log(df['Volume'] + 1)
    
    # Calculate True Range
    true_range = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - previous_close).abs(),
        (df['Low'] - previous_close).abs()
    ], axis=1).max(axis=1)
    
    # Calculate True Range Adjusted Volatility
    price_range = df[['Open', 'High', 'Low', 'Close']]
    volatility = price_range.rolling(window=20).std().mean(axis=1)
    true_range_adjusted_volatility = volatility / true_range
    
    # Calculate Smoothed Trend Indicator
    two_day_ma = df['Close'].rolling(window=2).mean()
    trend_indicator = df['Close'] - two_day_ma
    trend_indicator_smoothed = trend_indicator.rolling(window=5).sum()
    
    # Combine all components
    factor = (
        close_to_high_low_spread +
        volume_weighted_momentum +
        true_range_adjusted_volatility +
        trend_indicator_smoothed
    )
    
    return factor
