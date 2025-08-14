import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Calculate Adjusted High-Low Spread
    adjusted_high_low_spread = high_low_spread / df['close']
    
    # Calculate 10-day and 20-day Moving Averages
    ma_10 = df['close'].rolling(window=10).mean()
    ma_20 = df['close'].rolling(window=20).mean()
    
    # Calculate Price Momentum
    price_momentum = ma_20 - ma_10
    
    # Calculate 7-day Median Volume
    median_volume_7 = df['volume'].rolling(window=7).median()
    
    # Calculate Current Day Volume Ratio
    volume_ratio = df['volume'] / median_volume_7
    volume_weighted_momentum = adjusted_high_low_spread * (df['volume'] * volume_ratio)
    
    # Calculate Daily High-Low Difference and Intraday Volatility
    daily_high_low_diff = df['high'] - df['low']
    ema_high_low_diff = daily_high_low_diff.ewm(span=5).mean()
    high_low_momentum = daily_high_low_diff - ema_high_low_diff.shift(1)
    intraday_volatility = (df['high'] - df['low']) / df['open']
    combined_momentum_volatility = high_low_momentum * intraday_volatility
    
    # Smooth the Combined Factor
    sma_window_size = 5
    smoothed_combined_factor = combined_momentum_volatility.rolling(window=sma_window_size).mean()
    
    # Calculate Volume Adjusted Return
    short_term_return = df['close'].pct_change()
    long_term_return = df['close'].pct_change(periods=20)
    volume_adjusted_short_term = short_term_return * df['volume']
    volume_adjusted_long_term = long_term_return * df['volume']
    
    # Calculate Price Reversal Indicator
    local_highs = df['high'].rolling(window=5, center=True).max()
    local_lows = df['low'].rolling(window=5, center=True).min()
    time_since_high = (df.index - df[df['high'] == local_highs].index.to_series().reindex(df.index, method='ffill')).dt.days
    time_since_low = (df.index - df[df['low'] == local_lows].index.to_series().reindex(df.index, method='ffill')).dt.days
    price_reversal_indicator = (time_since_high - time_since_low) / 5  # Linear decay over 5 days
    
    # Combine Metrics
    factor = (volume_adjusted_short_term - volume_adjusted_long_term) + smoothed_combined_factor + price_reversal_indicator
    
    return factor
