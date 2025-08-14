import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Calculate Intraday Mid-Price
    mid_price = (df['high'] + df['low']) / 2
    
    # Calculate Open-to-Mid-Price Deviation
    open_mid_deviation = abs(df['open'] - mid_price)
    
    # Calculate Close-to-Mid-Price Deviation
    close_mid_deviation = abs(df['close'] - mid_price)
    
    # Combine Open and Close Deviations
    combined_deviation = 0.5 * open_mid_deviation + 0.5 * close_mid_deviation
    
    # Adjust for Volume and Amount
    volume_to_amount_ratio = df['volume'] / df['amount']
    adjusted_combined_deviation = combined_deviation * volume_to_amount_ratio
    
    # Enhance Intraday Price Changes
    high_low_diff = df['high'] - df['low']
    open_close_diff = df['open'] - df['close']
    intraday_price_change = high_low_diff + open_close_diff
    
    # Calculate Volume-Weighted Intraday Momentum
    avg_intraday_price = (df['high'] + df['low'] + df['open'] + df['close']) / 4
    volume_weighted_momentum = avg_intraday_price * df['volume']
    
    # Apply Exponential Moving Average
    ema_volume_weighted_momentum = volume_weighted_momentum.ewm(span=10, adjust=False).mean()
    
    # Introduce Additional Volume Impact
    volume_change_rate = df['volume'].pct_change().fillna(0)
    adjusted_volume_weighted_momentum = ema_volume_weighted_momentum * volume_change_rate
    
    # Calculate Daily High-Low Range
    daily_high_low_range = df['high'] - df['low']
    
    # Adjust High-Low Range by Volume
    average_volume_30_days = df['volume'].rolling(window=30).mean()
    adjusted_high_low_range = daily_high_low_range * (df['volume'] / average_volume_30_days)
    
    # Compute Weighted Sum of Recent Adjusted Ranges
    decay_factor = 0.9
    weighted_sum_adjusted_ranges = 0
    for i in range(1, 11):
        decay = decay_factor ** i
        weighted_sum_adjusted_ranges += adjusted_high_low_range.shift(i) * decay
    high_low_range_momentum = weighted_sum_adjusted_ranges.sum(axis=1)
    
    # Weight Combined Deviation by Volume
    weighted_combined_deviation = adjusted_combined_deviation * df['volume']
    
    # Synthesize Final Alpha Factor
    final_alpha_factor = (
        weighted_combined_deviation +
        ema_volume_weighted_momentum +
        adjusted_volume_weighted_momentum +
        high_low_range_momentum
    )
    
    return final_alpha_factor
