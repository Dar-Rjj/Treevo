import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday High-Low Spread
    intraday_volatility = df['high'] - df['low']
    
    # Logarithmic Return
    log_return = np.log(df['close'] / df['close'].shift(1))
    
    # Close-Open Spread
    close_open_spread = df['close'] - df['open']
    
    # Volume Weighted High-Low Spread
    volume_weighted_high_low = (df['high'] - df['low']) * df['volume']
    
    # Enhanced Liquidity Measure
    enhanced_liquidity = df['volume'] / df['amount']
    
    # 20-Day Moving Average of Close Price
    ma_20 = df['close'].rolling(window=20).mean()
    
    # Market Trend Adjustment
    market_trend = df['close'] - ma_20
    
    # Refined Momentum Calculation
    n = 20
    weights = df['volume'].rolling(window=n, min_periods=1).apply(lambda x: x / x.sum(), raw=False)
    refined_momentum = (log_return.rolling(window=n, min_periods=1) * weights).sum()
    
    # Combine Intraday Volatility, Momentum, Volume-Weighted, Enhanced Liquidity, and Market Trend Measures
    combined_measures = (intraday_volatility + refined_momentum + 
                         volume_weighted_high_low + enhanced_liquidity + 
                         market_trend)
    
    # Dynamic Gap Adjustment
    gap_difference = (df['open'] - df['close'].shift(1)).abs()
    mean_volume_20 = df['volume'].rolling(window=20, min_periods=1).mean()
    scaled_gap = gap_difference * (df['volume'] / mean_volume_20)
    
    # Final Alpha Factor
    final_alpha = combined_measures + scaled_gap
    
    return final_alpha
