import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate Intraday High-Low Spread
    intraday_high_low_spread = df['high'] - df['low']

    # Calculate Logarithmic Return
    log_returns = np.log(df['close'] / df['close'].shift(1))

    # Calculate Close-Open Spread
    close_open_spread = df['close'] - df['open']

    # Calculate Volume Weighted High-Low Spread
    volume_weighted_high_low_spread = (df['high'] - df['low']) * df['volume']

    # Refined Momentum Calculation
    rolling_log_return_mean = log_returns.rolling(window=5).mean()
    # Adjust for Volume Trends
    weighted_recent_log_returns = rolling_log_return_mean * (df['volume'] / df['volume'].rolling(window=5).mean())

    # Liquidity and Market Depth Analysis
    # Calculate Average True Range (ATR)
    true_range = np.maximum(df['high'] - df['low'], np.abs(df['close'].shift(1) - df['high']), np.abs(df['close'].shift(1) - df['low']))
    atr = true_range.rolling(window=14).mean()
    
    # Calculate Volume Moving Average
    volume_ma = df['volume'].rolling(window=10).mean()

    # Dynamic Gap Adjustment
    gap_difference = (df['open'] - df['close'].shift(1)).abs()
    scaled_gap_difference = gap_difference * (df['volume'] / volume_ma)

    # Enhanced Volume Trends Analysis
    # Calculate Volume Trend (Linear Regression Slope)
    def calculate_volume_trend(volume, window=5):
        trend = volume.rolling(window=window).apply(lambda x: linregress(np.arange(len(x)), x)[0], raw=False)
        return trend
    
    volume_trend = calculate_volume_trend(df['volume'])

    # Adjust Momentum for Volume Trend
    momentum_adjusted_for_volume_trend = weighted_recent_log_returns + volume_trend

    # Final Alpha Factor
    alpha_factor = (
        intraday_high_low_spread +
        momentum_adjusted_for_volume_trend +
        volume_weighted_high_low_spread +
        np.where(df['open'] > df['close'].shift(1), scaled_gap_difference, -scaled_gap_difference)
    )

    return alpha_factor
