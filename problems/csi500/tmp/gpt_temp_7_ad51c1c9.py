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
    ewm_close = df['close'].ewm(span=20).mean()
    market_trend_adjustment = df['close'] - ewm_close

    # Refined Momentum Calculation
    recent_log_returns = log_return.rolling(window=20, min_periods=1).apply(lambda x: (x * df['volume']).ewm(span=20).mean().iloc[-1], raw=False)
    refined_momentum = recent_log_returns

    # Combine Intraday Volatility, Momentum, Volume-Weighted, Enhanced Liquidity, and Market Trend Measures
    combined_measures = (
        intraday_high_low_spread +
        refined_momentum +
        volume_weighted_high_low_spread +
        enhanced_liquidity_measure +
        market_trend_adjustment
    )

    # Gap Adjustment
    gap_difference = (df['open'] - df['close'].shift(1)).abs()
    scaled_gap_difference = gap_difference * df['volume']
    
    # Final Alpha Factor
    final_alpha_factor = combined_measures + scaled_gap_difference

    return final_alpha_factor
