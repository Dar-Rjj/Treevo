import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Difference
    high_low_diff = df['high'] - df['low']
    
    # Compute Adjusted High-Low Range
    mid_price = (df['open'] + df['close']) / 2
    adj_high_low_range = (high_low_diff * mid_price) / df['volume'].rolling(window=5).mean()
    
    # Smoothing the Factor with Exponential Moving Average
    alpha_factor = adj_high_low_range.ewm(alpha=0.2, adjust=False).mean()
    
    # Consider basic price dynamics
    daily_return = df['close'].pct_change()
    intraday_movement = high_low_diff
    overnight_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    close_strength = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Incorporate volume analysis
    vol_weighted_return = (df['close'] - df['close'].shift(1)) * df['volume'] / df['close'].shift(1)
    vol_change = df['volume'].pct_change()
    avg_vol_ratio = df['volume'] / df['volume'].rolling(window=5).mean()
    
    # Examine trade amount
    trade_amount_per_unit = df['amount'] / df['close']
    trade_amount_trend = (df['amount'] - df['amount'].shift(5)) / df['amount'].shift(5)
    trade_amount_intraday_range = (df['amount'] / df['close']) * high_low_diff
    
    # Explore relationships between different metrics
    price_vol_corr = (df['close'] - df['close'].shift(5)).corrwith(df['volume'] - df['volume'].shift(5))
    intraday_close_ratio = high_low_diff / df['close']
    joint_effect = daily_return * vol_change * (high_low_diff / df['close'])
    
    # Combine all factors into a single alpha factor
    alpha_factor = (
        0.3 * alpha_factor +
        0.1 * daily_return +
        0.1 * intraday_movement +
        0.1 * overnight_gap +
        0.1 * close_strength +
        0.1 * vol_weighted_return +
        0.05 * vol_change +
        0.05 * avg_vol_ratio +
        0.05 * trade_amount_per_unit +
        0.05 * trade_amount_trend +
        0.05 * trade_amount_intraday_range +
        0.05 * price_vol_corr +
        0.05 * intraday_close_ratio +
        0.05 * joint_effect
    )
    
    return alpha_factor
