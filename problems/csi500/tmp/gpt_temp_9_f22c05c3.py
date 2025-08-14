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
    
    # 20-Day Moving Average of Close Price
    ma_20 = df['close'].rolling(window=20).mean()
    deviation_from_ma_20 = df['close'] - ma_20
    
    # Refined Momentum Calculation
    rolling_window = 10  # example window, can be adjusted
    recent_log_returns = log_return.rolling(window=rolling_window).apply(
        lambda x: np.average(x, weights=df.loc[x.index, 'volume']), raw=False)
    
    # Combine Intraday Volatility, Momentum, Volume-Weighted, Enhanced Liquidity, and Market Trend Measures
    combined_measures = (
        intraday_high_low_spread +
        recent_log_returns +
        volume_weighted_high_low_spread +
        enhanced_liquidity_measure +
        deviation_from_ma_20
    )
    
    # Dynamic Gap Adjustment
    gap_difference = (df['open'] - df['close'].shift(1)).abs()
    scaled_gap_difference = gap_difference * df['volume']
    gap_adjustment = np.where(df['open'] > df['close'].shift(1), 
                              scaled_gap_difference, 
                              -scaled_gap_difference)
    
    # Final Alpha Factor
    final_alpha_factor = combined_measures + gap_adjustment
    
    # Incorporate Volume Trends
    volume_growth_rate = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Incorporating the volume growth rate into the final alpha factor
    final_alpha_factor += volume_growth_rate
    
    return final_alpha_factor
