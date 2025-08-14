import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Intraday High-Low Spread
    intraday_high_low_spread = df['high'] - df['low']
    
    # Compute Previous Day's Close-to-Open Return
    prev_close_to_open_return = df['close'].shift(1) - df['open']
    
    # Calculate Volume Weighted Average Price (VWAP)
    prices = (df[['open', 'high', 'low', 'close']].sum(axis=1) / 4)
    vwap = (prices * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Combine Intraday Momentum and VWAP
    combined_intraday_momentum = vwap - intraday_high_low_spread
    weighted_intraday_momentum = combined_intraday_momentum * df['volume']
    
    # Incorporate Multi-Period Momentum
    ten_day_momentum = df['close'] - df['close'].shift(10)
    twenty_day_momentum = df['close'] - df['close'].shift(20)
    fifty_day_momentum = df['close'] - df['close'].shift(50)
    
    # Enhance with Long-Term Price Trend
    fifty_day_ma = df['close'].rolling(window=50).mean()
    
    # Integrate Short-Term Volatility
    def true_range(row):
        return max(row['high'] - row['low'], abs(row['high'] - row['close'].shift(1)), abs(row['low'] - row['close'].shift(1)))
    df['true_range'] = df.apply(true_range, axis=1)
    atr_14 = df['true_range'].rolling(window=14).mean()
    
    # Incorporate Short-Term Price Trend
    seven_day_ma = df['close'].rolling(window=7).mean()
    
    # Final Factor Combination
    alpha_factor = (
        weighted_intraday_momentum + 
        ten_day_momentum + 
        twenty_day_momentum + 
        fifty_day_momentum + 
        prev_close_to_open_return + 
        fifty_day_ma + 
        atr_14 + 
        seven_day_ma
    )
    
    # Smooth the Factor
    smoothed_alpha_factor = alpha_factor.ewm(span=20).mean()
    
    return smoothed_alpha_factor
