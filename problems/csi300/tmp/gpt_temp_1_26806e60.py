import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Difference
    high_low_diff = df['high'] - df['low']
    
    # Compute Intraday High-Low Ratio
    intraday_high_low_ratio = (df['high'] - df['low']) / df['low']
    
    # Calculate Close-to-Open Return
    close_to_open_return = (df['close'] - df['open']) / df['open']
    
    # Compute Volume Momentum
    lookback_period = 10
    avg_volume = df['volume'].rolling(window=lookback_period).mean()
    volume_momentum = df['volume'] * (df['volume'] / avg_volume)
    
    # Calculate Volume Weighted Return
    volume_weighted_return = df['volume'] * close_to_open_return
    
    # Combine High-Low Difference, Intraday High-Low Ratio, and Volume Momentum
    combined_factor = (high_low_diff * volume_momentum) + (intraday_high_low_ratio * volume_weighted_return)
    
    # Integrate VWAP into Alpha Factor
    vwap = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    vwap_momentum = vwap.rolling(window=5).sum()
    
    # Adjust Price Momentum
    ten_day_avg_return = df['close'].pct_change().rolling(window=10).mean()
    five_day_avg_volume = df['volume'].rolling(window=5).mean()
    one_day_vol_change = df['volume'] - five_day_avg_volume
    adjusted_price_momentum = ten_day_avg_return * (1.2 if one_day_vol_change < 0 else -1.2)
    
    # Calculate Volume-Adjusted High-Low Momentum
    volume_adjusted_high_low_momentum = (df['high'] - df['low']) * df['volume']
    rolling_cumulative_sum = volume_adjusted_high_low_momentum.rolling(window=5).sum()
    
    # Final Factor Combination
    final_factor = (adjusted_price_momentum + combined_factor + rolling_cumulative_sum).ewm(span=5).mean()
    
    return final_factor
