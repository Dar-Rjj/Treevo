import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, market_data):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Close to Open Ratio
    close_to_open_ratio = df['close'] / df['open']
    
    # Weighted Difference
    weighted_diff = (intraday_range * close_to_open_ratio) * df['volume']
    
    # Integrate Multi-Day Momentum
    five_day_return = df['close'] / df['close'].shift(5)
    ten_day_return = df['close'] / df['close'].shift(10)
    twenty_day_return = df['close'] / df['close'].shift(20)
    
    multi_day_momentum = (five_day_return + ten_day_return + twenty_day_return) / 3
    combined_returns = multi_day_momentum * weighted_diff
    
    # Incorporate Volatility
    daily_price_change = df['close'] - df['close'].shift(1)
    volatility = daily_price_change.rolling(window=20).std()
    adjusted_for_volatility = combined_returns / volatility
    
    # Sector Trend Adjustment
    sector_average_return = market_data.groupby('sector')['close'].pct_change(20).mean(level='sector')
    sector_trend_adjusted = adjusted_for_volatility - sector_average_return[df['sector']]
    
    # Market Cap Adjustment
    market_cap = df['close'] * market_data['shares_outstanding']
    final_factor = sector_trend_adjusted / market_cap
    
    return final_factor
