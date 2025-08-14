import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Difference
    high_low_diff = df['high'] - df['low']
    
    # Volume-Weight Adjusted High-Low Momentum
    vol_weight_high_low_mom = (high_low_diff * df['volume']).rolling(window=5).sum()
    
    # Incorporate Close-to-Open Return
    close_to_open_return = (df['close'] - df['open']) * (high_low_diff / df['volume'])
    
    # Calculate 10-Day Average Return
    avg_return_10d = df['close'].pct_change().rolling(window=10).mean()
    
    # Calculate Volume Reversal
    vol_5d_ma = df['volume'].rolling(window=5).mean()
    vol_1d_change = df['volume'] - vol_5d_ma
    vol_reversal = vol_1d_change.apply(lambda x: 1 if x > 0 else -1)
    
    # Adjust Price Momentum
    adj_price_momentum = avg_return_10d * vol_reversal
    
    # Calculate Momentum
    n = 5
    momentum = df['close'].diff(n)
    
    # Calculate Relative Strength
    pos_mom = momentum[momentum > 0].sum()
    neg_mom = momentum[momentum < 0].sum()
    
    # Compute Relative Strength Ratio
    rs_ratio = pos_mom / (-neg_mom)
    
    # Apply Smoothed Moving Average
    m = 10
    smoothed_rs_ratio = rs_ratio.ewm(span=m, min_periods=m).mean()
    
    # Incorporate VWAP into Alpha Factor
    vwap = (df[['open', 'high', 'low', 'close']].mean(axis=1) * df['volume']).cumsum() / df['volume'].cumsum()
    vwap_mom = vwap.diff().rolling(window=5).sum()
    
    # High-Low Range Momentum
    daily_range = df['high'] - df['low']
    range_mom = daily_range.diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    
    # Volume-Amount Ratio Trend
    vol_amt_ratio = df['volume'] / df['amount']
    sum_last_5_days = vol_amt_ratio.rolling(window=5).sum()
    trend = (vol_amt_ratio.rolling(window=5).sum() - sum_last_5_days.shift(5)).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    
    # Combine High-Low Range Momentum and Volume-Amount Ratio Trend
    combined_trend = range_mom * trend
    
    # Final Alpha Factor
    alpha_factor = (vol_weight_high_low_mom + close_to_open_return + vwap_mom + adj_price_momentum + combined_trend).fillna(0)
    
    return alpha_factor
