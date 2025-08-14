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
    market_trend_adjustment = df['close'] - ma_20
    
    # Refined Momentum Calculation
    weights = df['volume'].rolling(window=20).apply(lambda x: x / x.sum(), raw=False)
    weighted_log_returns = (log_return.rolling(window=20) * weights).sum()
    
    # Combine measures
    combined_measures = (
        intraday_volatility + 
        weighted_log_returns + 
        volume_weighted_high_low + 
        enhanced_liquidity + 
        market_trend_adjustment
    )
    
    # Dynamic Gap Adjustment
    gap_difference = (df['open'] - df['close'].shift(1)).abs()
    adjusted_gap_difference = gap_difference * df['volume']
    
    # Final Alpha Factor
    final_alpha_factor = combined_measures + adjusted_gap_difference
    
    # Incorporate Volume Trends
    volume_growth_rate = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    final_alpha_factor = final_alpha_factor * (1 + volume_growth_rate)
    
    return final_alpha_factor
