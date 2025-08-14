import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, short_period=5, long_period=20, decay_factor=0.85, ema_span=10, intraday_volatility_days=10, breakout_days=30):
    # Calculate Daily High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Compute Weighted Sum of Recent Spreads
    cumulative_decay = np.cumprod([decay_factor] * (short_period - 1))
    spread_momentum = (high_low_range.rolling(window=short_period).apply(lambda x: np.sum(x * cumulative_decay), raw=True)).fillna(0)
    
    # Compute Volume-Weighted High-Low Range Over Short Period
    volume_weighted_short = (high_low_range * df['volume']).rolling(window=short_period).sum()
    
    # Compute Volume-Weighted High-Low Range Over Long Period
    volume_weighted_long = (high_low_range * df['volume']).rolling(window=long_period).sum()
    
    # Compute the High-Low Range Momentum
    high_low_range_momentum = volume_weighted_short / volume_weighted_long
    
    # Adjusted Momentum Calculation
    ema_close = df['close'].ewm(span=ema_span, adjust=False).mean()
    ema_return = ema_close.diff().fillna(0)
    intraday_range = df['high'] - df['low']
    average_intraday_range = intraday_range.rolling(window=intraday_volatility_days).mean()
    volatility_adjusted_momentum = ema_return / average_intraday_range
    
    # Calculate Breakout Potential
    max_high = df['high'].rolling(window=breakout_days).max()
    min_low = df['low'].rolling(window=breakout_days).min()
    breakout_range = max_high - min_low
    breakout_ratio = breakout_range / df['close']
    
    # Combine Momentum Indicators and Breakout Potential
    combined_momentum = (high_low_range_momentum + volatility_adjusted_momentum + breakout_ratio) * spread_momentum
    
    # Incorporate Trend Strength
    trend_direction = df['close'].ewm(span=ema_span, adjust=False).mean() - df['close'].ewm(span=ema_span + 5, adjust=False).mean()
    trend_enhanced_momentum = combined_momentum * trend_direction
    
    return trend_enhanced_momentum

# Example usage:
# df = pd.read_csv('your_stock_data.csv', parse_dates=['date'], index_col='date')
# factor = heuristics_v2(df)
# print(factor)
