import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Adjust Intraday Range by Volume
    ema_volume = df['volume'].ewm(span=20).mean()
    adjusted_volume = df['volume'] / ema_volume
    adjusted_intraday_range = intraday_range * adjusted_volume
    
    # Further Adjustment by Close Price Volatility
    close_returns = df['close'].pct_change()
    close_volatility = close_returns.rolling(window=20).std()
    adjusted_intraday_range_volatility = adjusted_intraday_range / close_volatility
    
    # Calculate True Range Using High, Low, and Close Prices
    true_range = np.maximum.reduce([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ])
    
    # Adjust Intraday Range by True Range Volatility
    true_range_volatility = true_range.rolling(window=20).std()
    adjusted_intraday_range_true_range = adjusted_intraday_range / true_range_volatility
    
    # Calculate Daily High-Low Difference
    high_low_diff = df['high'] - df['low']
    
    # Compute Exponential Moving Average (EMA) of High-Low Difference
    ema_high_low = high_low_diff.ewm(span=20).mean()
    
    # Compute High-Low Momentum
    high_low_momentum = high_low_diff - ema_high_low.shift(1)
    
    # Calculate Price Momentum
    ma_10 = df['close'].rolling(window=10).mean()
    ma_20 = df['close'].rolling(window=20).mean()
    price_momentum = ma_10 - ma_20
    
    # Calculate Volume Spike
    median_volume_7 = df['volume'].rolling(window=7).median()
    volume_ratio = df['volume'] / median_volume_7
    
    # Enhance Price-Velocity Factors
    roc_5 = df['close'].pct_change(5)
    roc_10 = df['close'].pct_change(10)
    roc_3_volume = df['volume'].pct_change(3)
    roc_7_volume = df['volume'].pct_change(7)
    
    # Synthesize Intraday, High-Low, and Price-Volume Momentum
    intraday_adjusted_volume_ratio = adjusted_intraday_range * volume_ratio
    intraday_adjusted_true_range = adjusted_intraday_range_true_range * volume_ratio
    high_low_momentum_volume_ratio = high_low_momentum * volume_ratio
    price_momentum_volume_ratio = price_momentum * volume_ratio
    price_velocity_volume_ratio = (roc_5 + roc_10) * volume_ratio
    volume_velocity_volume_ratio = (roc_3_volume + roc_7_volume) * volume_ratio
    
    # Sum the Six Momentum Components
    alpha_factor = (
        intraday_adjusted_volume_ratio +
        intraday_adjusted_true_range +
        high_low_momentum_volume_ratio +
        price_momentum_volume_ratio +
        price_velocity_volume_ratio +
        volume_velocity_volume_ratio
    )
    
    return alpha_factor
