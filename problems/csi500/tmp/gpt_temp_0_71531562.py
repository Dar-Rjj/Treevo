import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday High-Low Spread
    intraday_high_low_spread = df['high'] - df['low']
    
    # Logarithmic Return
    log_return = np.log(df['close'] / df['close'].shift(1))
    
    # Close-Open Spread
    close_open_spread = df['close'] - df['open']
    
    # Volume Weighted High-Low Spread
    volume_weighted_high_low_spread = (df['high'] - df['low']) * df['volume']
    
    # Enhanced Liquidity Measure
    enhanced_liquidity_measure = df['volume'] / df['amount']
    
    # Market Trend Adjustment
    ma_20 = df['close'].rolling(window=20).mean()
    market_trend_adjustment = df['close'] - ma_20
    
    # Refined Momentum Calculation
    momentum = log_return.rolling(window=10).apply(lambda x: np.average(x, weights=df['volume']))
    
    # Combine Intraday Volatility, Momentum, Volume-Weighted, Enhanced Liquidity, and Market Trend Measures
    combined_measures = (
        intraday_high_low_spread +
        momentum +
        volume_weighted_high_low_spread +
        enhanced_liquidity_measure +
        market_trend_adjustment
    )
    
    # Dynamic Gap Adjustment
    gap_difference = df['open'] - df['close'].shift(1)
    scaled_gap_difference = gap_difference * df['volume']
    
    # Final Alpha Factor
    final_alpha_factor = combined_measures + scaled_gap_difference
    
    return final_alpha_factor
