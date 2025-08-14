import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Price Difference
    high_low_diff = df['high'] - df['low']
    
    # Calculate 5-Day Average Volume
    avg_volume_5d = df['volume'].rolling(window=5).mean()
    
    # Adjust High-Low Difference by Average Volume
    adjusted_high_low = high_low_diff / avg_volume_5d
    
    # Calculate 20-Day Standard Deviation of Adjusted High-Low
    std_dev_adjusted_high_low = adjusted_high_low.rolling(window=20).std()
    
    # Calculate Daily Momentum
    daily_momentum = df['close'] - df['close'].shift(20)
    
    # Calculate Average Volume over 20 Days
    avg_volume_20d = df['volume'].rolling(window=20).mean()
    
    # Calculate Volume Weighted Price
    volume_weighted_price = df['close'] * avg_volume_20d
    
    # Calculate Price Momentum Ratio
    price_momentum_ratio = daily_momentum / volume_weighted_price
    
    # Calculate Intraday Return
    intraday_return = (df['close'] - df['open']) / df['open']
    
    # Calculate Next Day Overnight Return
    overnight_return = (df['open'].shift(-1) - df['close']) / df['close']
    
    # Combine Intraday and Overnight Returns
    combined_return = (overnight_return - intraday_return) * df['volume']
    
    # Determine Market Activity Level
    avg_volume = df['volume'].expanding().mean()
    market_activity_level = avg_volume.rolling(window=20).mean() / avg_volume
    
    # Adjust Lookback Period Based on Market Activity
    def dynamic_window(row):
        if row > 1:
            return 10
        else:
            return 30
    lookback_period = market_activity_level.apply(dynamic_window)
    
    # Calculate Rolling Sum of Volumetric Weighted Returns
    rolling_sum_combined_return = combined_return.rolling(window=lookback_period, min_periods=10).sum()
    
    # Integrate Factors
    factor = std_dev_adjusted_high_low * price_momentum_ratio + rolling_sum_combined_return
    
    return factor
